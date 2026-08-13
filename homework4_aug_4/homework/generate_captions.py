from pathlib import Path
import json

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info, _relative_position


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position
    # {kart_name} is {position} of the ego car.

    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)
    captions = []

    if not kart_objects:
        return [f"The track is {track_name}."]

    ego = next((kart for kart in kart_objects if kart["is_center_kart"]), kart_objects[0])
    other_karts = [kart for kart in kart_objects if kart is not ego]

    captions.append(f"{ego['kart_name']} is the ego car.")
    captions.append(f"There are {len(kart_objects)} karts in the scene.")
    captions.append(f"The track is {track_name}.")

    for kart in other_karts:
        _, _, position = _relative_position(kart, ego)
        captions.append(f"{kart['kart_name']} is {position} of the ego car.")

    return captions


def generate_dataset(data_dir: str = "data", split: str = "train", output_file: str | None = None):
    data_path = Path(data_dir)
    split_path = data_path / split
    captions = []

    for info_file in sorted(split_path.glob("*_info.json")):
        with open(info_file) as f:
            info = json.load(f)

        base_name = info_file.stem.replace("_info", "")
        for view_index in range(len(info["detections"])):
            image_file = split_path / f"{base_name}_{view_index:02d}_im.jpg"
            if not image_file.exists():
                continue

            for caption in generate_caption(str(info_file), view_index):
                captions.append(
                    {
                        "image_file": f"{split}/{image_file.name}",
                        "caption": caption,
                    }
                )

    if output_file is None:
        output_file = split_path / "generated_captions.json"
    else:
        output_file = Path(output_file)

    with output_file.open("w") as f:
        json.dump(captions, f, indent=2)

    print(f"Wrote {len(captions)} captions to {output_file}")


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption, "generate": generate_dataset})


if __name__ == "__main__":
    main()
    