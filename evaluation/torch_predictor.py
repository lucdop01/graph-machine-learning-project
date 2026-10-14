"""Rollouts with a PyTorch Geometric model (plan step 3), and verified transfers between the CPU and a GPU.

TorchPredictor wraps a model that maps a GraphSample (graphs/dataset.py SampleBuilder, the same features as in
training) to the normalised one-step change of its target, one value per node:
  target "dT"   T_next = T + dT                       (GCN, models A and B)
  target "dH"   T_next = T(H(T) + dH), material table  (models C and D; D computes dH from its edge fluxes inside
                                                        the model)
The change is de-normalised with the training statistics the model was trained with (stats["target"]). The statistics
must fit each run (stride, interpolation, frame interval); load them with graphs.dataset.load_stats, which also checks
them against the current training data, or take them from the model's checkpoint.
Each step builds the input from the predictor's own previous output, so no stored truth reaches the model.
reset() switches the model to evaluation mode before every rollout, because training code switches it back to training
mode between validations (models/train.py). The model stays in evaluation mode after the rollout; the caller restores
training mode.

Verified transfers. ROCm on native Windows once returned the previous device-to-host read instead of the new value
(models/train.py stopped on a "non-finite" step whose number was the bit pattern of the previous log row's gradient
norm). So every transfer between the CPU and a GPU that a result depends on is checked:
  to_host()          a read from the GPU. The copy carries a fresh marker at both ends (made on the GPU, so the
                     marker is not uploaded itself) and an exact checksum of its contents computed on the GPU, plus
                     the marker (so a checksum left over from an earlier read does not fit). A copy with a wrong
                     marker or checksum is read again after a synchronise, each time with a new marker
  check_on_device()  tensors uploaded to the GPU (weights, Adam state, mesh tensors) against their CPU source, by
                     exact checksums read with to_host(); a tensor that differs is uploaded again. move_checked()
                     moves a model this way (parameters and buffers)
  TorchPredictor     on a GPU, each step's input is built on the CPU and uploaded; its checksum on the GPU comes back
                     in the same verified read as the prediction, and the step is repeated if it differs
bit_checksum() gives two sums of the bytes, each byte weighted by a pseudo-random 24-bit weight of its position (two
multiplicative hashes), in int64: exact, so they do not depend on the order of the summation and are equal on the CPU
and the GPU. Random weights matter: with weights that grow by one per byte, two nearby numbers often have the same
sum (a test caught a stale value that way). After READ_TRIES failed tries a transfer raises StaleReadError. Every
repeated transfer prints a warning and is counted in `retries`. On a CPU nothing is checked and everything runs as
before, bit for bit.
"""
import itertools
import os

import numpy as np

from graphs.dataset import TARGETS, GraphSample, SampleBuilder, check_stats, check_stride, require_torch

try:
    import torch
except ModuleNotFoundError as e:                     # the error is raised in TorchPredictor (require_torch)
    if e.name != "torch":
        raise

READ_TRIES = 3                                       # tries per transfer before StaleReadError
CHECKSUM_CHUNK = 1 << 22                             # bytes per checksum pass; the weights repeat with this period
CHECKSUM_MAX = 1 << 30                               # bytes per checksum: the sums stay below 2^62
_HASH = (2654435761, 2246822519)                     # odd 32-bit multipliers of the two position hashes
_weights = {}                                        # device -> int64 weights (2, n), made on that device
_markers = itertools.count(int.from_bytes(os.urandom(5), "little"))       # < 2^41; another process's markers differ
retries = 0                                          # transfers repeated so far in this process (each one is printed)


class StaleReadError(RuntimeError):
    """A transfer between the CPU and a GPU still gave wrong data after READ_TRIES tries."""


# The functions below are the only places that move data between the CPU and a GPU; the tests replace them to
# simulate stale or damaged transfers on a CPU.

def needs_verify(device):
    """Transfers to and from this device are checked (every device but the CPU)."""
    return torch.device(device).type != "cpu"


def _copy(buf):
    """Device-to-host copy."""
    return buf.to("cpu")


def _upload(t, device):
    """Host-to-device copy."""
    return t.to(device)


def _recopy(dst, src):
    """Upload src again into the device tensor dst."""
    with torch.no_grad():
        dst.copy_(src)


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _as_bytes(t):
    b = t.detach().reshape(-1)
    if b.numel() and b.stride(0) != 1:               # e.g. one element of a strided view
        b = b.clone(memory_format=torch.contiguous_format)
    return b.view(torch.uint8)


def _repeat(what, attempt, device):
    """Count and report a transfer that is repeated, then synchronise before the next try."""
    global retries
    retries += 1
    print(f"WARNING: {what} was stale or damaged (try {attempt + 1} of {READ_TRIES}); repeated", flush=True)
    _sync(device)


def _position_weights(device, n):
    """Weights w_k(i) = ((i x HASH_k) mod 2^32) / 2^8 + 1, 1 .. 2^24, for i = 0 .. n - 1, computed on the device
    (no upload) and kept for the next call."""
    key = str(device)
    w = _weights.get(key)
    if w is None or w.shape[1] < n:
        i = torch.arange(min(CHECKSUM_CHUNK, max(n, 2 * (0 if w is None else w.shape[1]))), dtype=torch.int64,
                         device=device)
        w = torch.stack([((i * h) & 0xFFFFFFFF) // 256 + 1 for h in _HASH])
        _weights[key] = w
    return w[:, :n]


def bit_checksum(t):
    """Exact checksum of a tensor's bytes, as an int64 tensor of 2 values on the tensor's device: for k = 1, 2 the sum
    over the bytes b_i of b_i x w_k(i mod CHECKSUM_CHUNK), w_k pseudo-random weights of 1 .. 2^24. Integer
    arithmetic, so it does not depend on the order of the summation."""
    b = _as_bytes(t)
    if b.numel() > CHECKSUM_MAX:
        raise ValueError(f"bit_checksum takes at most {CHECKSUM_MAX} bytes, not {b.numel()}")
    total = torch.zeros(2, dtype=torch.int64, device=b.device)
    for start in range(0, b.numel(), CHECKSUM_CHUNK):
        c = b[start:start + CHECKSUM_CHUNK].to(torch.int64)
        total = total + (c * _position_weights(b.device, c.numel())).sum(dim=1)
    return total


def _int64s_at(host, pos, k=1):
    return host[pos:pos + 8 * k].clone().view(torch.int64).tolist()


def to_host(x, extra=None):
    """x (detached) on the CPU. From a GPU the copy is verified (module docstring). extra: an optional int64 tensor on
    the same device, read in the same verified copy; then (x on the CPU, extra as a list of ints) is returned."""
    x = x.detach()
    if not needs_verify(x.device):
        host = x.cpu()
        return host if extra is None else (host, [int(v) for v in extra.detach().cpu().reshape(-1)])
    ext = (extra.detach().reshape(-1).to(torch.int64) if extra is not None
           else torch.zeros(0, dtype=torch.int64, device=x.device))
    m = ext.numel()
    body = torch.cat([ext.view(torch.uint8), _as_bytes(x)])          # extra first, so x starts 8-byte aligned
    checksum = bit_checksum(body)
    for attempt in range(READ_TRIES):                # copy: [marker, checksums + marker, body, marker]
        marker = next(_markers)
        mark = torch.full((1,), marker, dtype=torch.int64, device=x.device).view(torch.uint8)
        salted = (checksum + marker).view(torch.uint8)               # an old checksum carries an old marker
        host = _copy(torch.cat([mark, salted, body, mark]))
        n = host.numel()
        if (n == body.numel() + 32 and _int64s_at(host, 0) == [marker] and _int64s_at(host, n - 8) == [marker]
                and _int64s_at(host, 8, 2) == [s + marker for s in bit_checksum(host[24:n - 8]).tolist()]):
            break
        _repeat(f"a read of {tuple(x.shape)} {x.dtype} from {x.device}", attempt, x.device)
    else:
        raise StaleReadError(f"a read of {tuple(x.shape)} {x.dtype} from {x.device} was stale or damaged in "
                             f"{READ_TRIES} tries")
    out = host[24 + 8 * m:n - 8].view(x.dtype).reshape(x.shape).clone()   # clone: its own compact storage
    if extra is None:
        return out
    return out, _int64s_at(host, 24, m)


def tree_to_host(obj):
    """to_host() for every tensor in nested dicts, lists and tuples (an optimiser's state_dict)."""
    if torch.is_tensor(obj):
        return to_host(obj)
    if isinstance(obj, dict):
        return {k: tree_to_host(v) for k, v in obj.items()}
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):           # a namedtuple
        return type(obj)(*(tree_to_host(v) for v in obj))
    if isinstance(obj, (list, tuple)):
        return type(obj)(tree_to_host(v) for v in obj)
    return obj


def check_on_device(on_device, source, what):
    """Check tensors uploaded to a GPU against their CPU source by exact checksums, in one verified read. on_device
    holds every key of source. A tensor that differs is uploaded again in place, up to READ_TRIES times; then
    StaleReadError. Tensors that are on the CPU are skipped."""
    keys = [k for k in source if needs_verify(on_device[k].device)]
    if not keys:
        return
    for k in keys:
        d, s = on_device[k], source[k]
        if s.device.type != "cpu":
            raise ValueError(f"{what}: the source of {k} must be on the CPU, not {s.device}")
        if d.dtype != s.dtype or d.shape != s.shape:
            raise ValueError(f"{what}: {k} is {d.dtype} {tuple(d.shape)} on the device but {s.dtype} "
                             f"{tuple(s.shape)} in its source")
    want = [bit_checksum(source[k]).tolist() for k in keys]
    device = on_device[keys[0]].device
    for attempt in range(READ_TRIES):
        got = to_host(torch.stack([bit_checksum(on_device[k]) for k in keys])).tolist()
        bad = [k for k, g, w in zip(keys, got, want) if g != w]
        if not bad:
            return
        for k in bad:
            _recopy(on_device[k], source[k])
        _repeat(f"{what}: the upload of {len(bad)} tensors (first: {bad[0]})", attempt, device)
    raise StaleReadError(f"{what}: {len(bad)} of {len(keys)} tensors on {device} still differ from their source after "
                         f"{READ_TRIES} uploads (first: {bad[0]})")


def move_checked(model, device, what):
    """model.to(device). On a GPU every parameter and buffer that was on the CPU is then checked against its CPU
    value (check_on_device), so a damaged upload is repaired or stops the program. Returns the model."""
    def tensors():
        return {**dict(model.named_parameters()), **dict(model.named_buffers())}
    source = ({k: v.detach().clone() for k, v in tensors().items() if v.device.type == "cpu"}
              if needs_verify(device) else {})
    model = model.to(device)
    if source:
        check_on_device(tensors(), source, what)
    return model


class TorchPredictor:
    """Rollout predictor for a model that outputs the normalised dT or dH per node."""

    def __init__(self, model, stats, stride, target="dT", device="cpu", name=None):
        require_torch("TorchPredictor")
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, not {target!r}")
        self.stats, self.stride, self.target = stats, check_stride(stride), target
        self.device = torch.device(device)
        self.verify = needs_verify(self.device)
        self.model = move_checked(model, self.device, f"weights of {type(model).__name__}").eval()
        # On a GPU the samples are built on the CPU and uploaded in step(), so that every upload can be checked
        self.builder = SampleBuilder(stats, self.stride, None if self.verify else self.device)
        self._mesh = {}                              # mesh tensors on the device, checked once per mesh
        self.mean, self.std = float(stats["target"][target]["mean"]), float(stats["target"][target]["std"])
        set_stats_fn = getattr(self.model, "set_stats", None)
        if callable(set_stats_fn):
            set_stats_fn(self.mean, self.std)
        self.name = name or f"{type(model).__name__}_{target}"

    def reset(self, run):
        check_stats(self.stats, self.stride, [run], f"run {run.name}")
        self.model.eval()

    def _upload(self, run, g):
        """The CPU sample g on the device: the mesh tensors from the cache, the step's own tensors uploaded now.
        Returns the sample, the checksum of the step's tensors computed on the device and the one on the CPU."""
        mesh = self.builder.mesh_tensors(run)
        key = (run.mesh.name, run.mesh.seed)
        if key not in self._mesh:
            dev = {k: _upload(v, self.device) for k, v in mesh.items()}
            check_on_device(dev, mesh, f"mesh tensors of {run.name}")
            self._mesh[key] = dev
        d = g.to_dict()
        own = [k for k, v in d.items() if torch.is_tensor(v) and k not in mesh]
        moved = {k: _upload(d[k], self.device) for k in own}
        rest = {k: v for k, v in d.items() if not torch.is_tensor(v)}
        sample = GraphSample(**self._mesh[key], **moved, **rest)
        on_device = bit_checksum(torch.cat([_as_bytes(moved[k]) for k in own]))
        on_cpu = bit_checksum(torch.cat([_as_bytes(d[k]) for k in own])).tolist()
        return sample, on_device, on_cpu

    def _output(self, run, k, T):
        """The model output for the step from frame k as a float64 array, from verified transfers on a GPU."""
        g = self.builder.sample(run, k, T)
        if not self.verify:
            with torch.no_grad():
                out = self.model(g)
            return out.detach().reshape(-1).to("cpu", torch.float64).numpy(), out.shape
        for attempt in range(READ_TRIES):
            sample, on_device, on_cpu = self._upload(run, g)
            with torch.no_grad():
                out = self.model(sample)
            z, uploaded = to_host(out.reshape(-1), extra=on_device)
            if uploaded == on_cpu:
                return z.to(torch.float64).numpy(), out.shape
            _repeat(f"{self.name}: the input of the step from frame {k} of {run.name}", attempt, self.device)
        raise StaleReadError(f"{self.name}: the input of the step from frame {k} of {run.name} reached "
                             f"{self.device} damaged in {READ_TRIES} uploads")

    def step(self, run, T, k, stride):
        if stride != self.stride:
            raise ValueError(f"{self.name} was built for stride {self.stride}, not {stride}")
        z, shape = self._output(run, k, T)
        if z.shape != (run.mesh.n_nodes,):
            raise ValueError(f"{self.name} returned {tuple(shape)}, expected one value per node "
                             f"({run.mesh.n_nodes})")
        if not np.isfinite(z).all():                 # stop at the step that produced it, with its run and frame
            raise FloatingPointError(f"{self.name} returned a non-finite output for the step from frame {k} of "
                                     f"{run.name}")
        delta = z * self.std + self.mean
        T = np.asarray(T, dtype=np.float64)
        if self.target == "dT":
            return T + delta
        return run.T_from_H(run.H_from_T(T) + delta)
