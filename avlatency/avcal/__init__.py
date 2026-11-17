"""Audio/video output latency calibration toolkit.

Modules:
  ffmpeg    - real media encode/decode through a bundled or system ffmpeg
  media     - generation of numbered flash frames and indexed audio pulses
  chain     - controllable recording chain (fixed delays, noise, drops, fps,
              clock drift, restart delay steps, single-event mis-matches)
  detect    - sample-resolution pulse/flash detection with explicit uncertainty
  match     - number/sequence based event alignment (no index pairing)
  drift     - piecewise drift/jump analysis: interpretable segments, audited
              outlier exclusion, independent holdout validation
  calibrate - latency kernel: offsets, uncertainty, verdicts, delay correction
  web       - multi-client control panel served over HTTP
"""

__version__ = "1.0.0"
