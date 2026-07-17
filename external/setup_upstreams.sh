#!/usr/bin/env bash
# Clone the upstream repositories required by external baseline wrappers and
# checkout the exact commits used during integration. Environments, checkpoints,
# and datasets are installed separately per package README.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

clone_at() {
  local url="$1" dest="$2" commit="$3"
  if [[ ! -d "$dest/.git" ]]; then
    git clone "$url" "$dest"
  fi
  git -C "$dest" fetch --depth 1 origin "$commit"
  git -C "$dest" checkout --detach "$commit"
  printf 'ready %-55s %s\n' "${dest#"$ROOT/"}" "$commit"
}

clone_at https://github.com/sfu-db/CleanAgent.git \
  "$ROOT/external/cleanagent/CleanAgent" \
  66c10c2c1d141e7d63946f80a3f660e072d5bbe0
clone_at https://github.com/Cocoon-Data-Transformation/cocoon.git \
  "$ROOT/external/cocoon/cocoon_repo" \
  410a4f710a93143bdf1d1b2c5c502cbb5a1d43dd
clone_at https://github.com/HazyResearch/fm_data_tasks.git \
  "$ROOT/external/fm_data_tasks/fm_data_tasks_repo" \
  c954cfdac6a420f773553e19dbfda81fba25df14
clone_at https://github.com/mengqiwang1/ForestED.git \
  "$ROOT/external/forested/ForestED" \
  6696c8ead73ecd58bf1468a44f21894f5be6c33e
clone_at https://github.com/Wangyibo321/TabClean.git \
  "$ROOT/external/tabclean/TabClean" \
  f514ce5ae5a914f21164e7eec2103216e6c3c852
clone_at https://github.com/WelkinNi/Automatic-Data-Repair.git \
  "$ROOT/external/adr/adr_repo" \
  b3004a8980104ae5412b8dc39e609b2c1c8e439d
clone_at https://github.com/HoloClean/holoclean.git \
  "$ROOT/external/holoclean/holoclean_repo" \
  d4f5929a8e4d92d4f41eb058c04c96cdcb0af767
clone_at https://github.com/BigDaMa/ExampleDrivenErrorDetection.git \
  "$ROOT/external/ed2/SequentialPatternErrorDetection" \
  ae8bc24fc441957d9a29e5fa4cc247f1805d8b4d
clone_at https://github.com/chu-data-lab/zeroer.git \
  "$ROOT/external/zeroer/zeroer" \
  4c1ea6acd2c214b32b2e52cca9b8d50afc180220

printf '\nRead external/README.md and each package README before installing or running a method.\n'