import os
import warnings

warnings.filterwarnings("ignore")

from config import ReproduceScissorConfig
from inference import OfficialScissorRunner


def main() -> None:
    video_path = "/data/zyk_data/LLaVA-Scissor/vedio1.mp4"
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    config = ReproduceScissorConfig(
        max_frames=4,
        max_new_tokens=16,
        seed=123,
    )
    runner = OfficialScissorRunner(config)
    result = runner.generate(video_path, "Describe briefly.")

    assert result.frame_shape[0] == 4
    assert result.text.strip()

    print("smoke_test_ok")
    print(f"frames={result.frame_shape}")
    print(f"elapsed_s={result.elapsed_s:.2f}")
    print(result.text)


if __name__ == "__main__":
    main()
