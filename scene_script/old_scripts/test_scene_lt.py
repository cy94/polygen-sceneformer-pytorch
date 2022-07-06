import torch
import os.path as osp
from models.scene_transformer import scene_transformer
from tqdm import tqdm
import argparse
from utils.config import read_config
from transforms.scene import Seq_to_Scene
from PIL import Image, ImageDraw, ImageFont
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


def img_gen(dic, index):
    image = Image.new("RGB", (2000, 2000), (255, 255, 255))
    font_type = ImageFont.truetype(
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf", size=30
    )
    draw = ImageDraw.Draw(image)
    for i in range(len(dic)):
        loc_0 = (dic[i]["loc"][0] * 10, dic[i]["loc"][2] * 10)
        cat_0 = dic[i]["class"]
        draw.text(xy=loc_0, text=cat_0, fill=(255, 69, 0), font=font_type)
    image.save(f"tests/data/scene_out/{index}.png")

    transform1 = transforms.Compose(
        [transforms.ToTensor(),]  # range [0, 255] -> [0.0,1.0]
    )
    image = transform1(image)
    return image


def run_inference(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = osp.join(cfg["test"]["model_file"])

    state_dict = torch.load(model_path)["state_dict"]
    model = scene_transformer(cfg)
    model.load_state_dict(state_dict)

    print(f"Using model: {model_path}")

    model = model  # .to(device)
    model.eval()

    num_valid = 0
    log_dir = cfg["test"]["log_dir"]
    writer = SummaryWriter(log_dir)
    with torch.no_grad():
        for out_ndx in tqdm(range(cfg["test"]["num_samples"])):
            seq = model.greedy_decode(probabilistic=cfg["test"]["probabilistic"])
            s2s = Seq_to_Scene()
            dic = s2s(seq)
            if dic is not False:
                image = img_gen(dic, out_ndx)
                writer.add_image("scene", image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", help="Path to config file")
    args = parser.parse_args()
    cfg = read_config(args.cfg_path)

    run_inference(cfg)
