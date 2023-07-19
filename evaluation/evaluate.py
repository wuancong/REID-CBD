import torchreid
from torchreid.evaluators import Evaluator
from torchreid.utils.preprocessor import Preprocessor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import argparse
from evaluate_utils import prepare_test_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="evaluation")
    parser.add_argument("--checkpoint_path", type=str, default='/path/to/checkpoint',
                        help="checkpoint path")
    parser.add_argument("--dataset_dir", type=str, default='/path/to/dataset',
                        help="dataset directory")
    parser.add_argument("--query_list", type=str, default='/path/to/query.txt',
                        help="query list")
    parser.add_argument("--gallery_list", type=str, default='/path/to/gallery.txt',
                        help="gallery list")
    parser.add_argument("--dataset_name", type=str, default='REID-CBD', choices=('REID-CBD', 'DukeMTMC', 'MSMT17'),
                        help="dataset name")
    args = parser.parse_args()

    model = torchreid.models.build_model(
        name='resnet50_fc512',
        num_classes=2675,
        loss='triplet',
        pretrained=True,
        checkpoint_path=args.checkpoint_path
    )

    model = model.cuda()

    query_data = prepare_test_data(args.dataset_name, args.query_list)
    gallery_data = prepare_test_data(args.dataset_name, args.gallery_list)

    transforms_test = transforms.Compose([
        transforms.Resize((256, 128), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    query_loader = DataLoader(Preprocessor(query_data, transform=transforms_test),
                              batch_size=256, shuffle=True, num_workers=4, drop_last=False)
    gallery_loader = DataLoader(Preprocessor(gallery_data, transform=transforms_test),
                                batch_size=256, shuffle=True, num_workers=4, drop_last=False)

    evaluator = Evaluator(model)
    results = evaluator.evaluate(query_loader, gallery_loader, query_data, gallery_data, cmc_flag=True)
