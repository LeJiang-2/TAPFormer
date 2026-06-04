import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble_task_multiscale import AnomalyCLIP_PromptLearner
from prompt_ensemble_clipad import encode_text_with_prompt_ensemble

from loss import FocalLoss, BinaryDiceLoss
from utils import normalize
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm

import os
import random
import numpy as np
from tabulate import tabulate
from utils import get_transform
from sklearn.metrics import pairwise
from scipy import ndimage

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import clip

mvtec_class = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
    ]
visa_class = [
        'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
        'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
        'pcb4', 'pipe_fryum',
    ]
mpdd_class = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes' ]
btad_class = ['01', '02', '03']
sdd_class = ['SDD']
dagm_class = ['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10']
dtd_class = ['Woven_001', 'Woven_127', 'Woven_104', 'Stratified_154', 'Blotchy_099', 'Woven_068', 'Woven_125', 'Marbled_078', 'Perforated_037', 'Mesh_114', 'Fibrous_183', 'Matted_069']
headct_class = ['headct']
brainmri_class = ['brain_mri']
br35h_class = ['br35h']
isic_class = ['isic']
clinicdb_class = ['ClinicDB']
colondb_class = ['ColonDB']
tn3k_class = ['tn3k']



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from visualization import visualizer

from metrics_kshot_corssattention_promptad_inctrl import image_level_metrics, pixel_level_metrics
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
def test(args):
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset

    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
    
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details = AnomalyCLIP_parameters)
    model.eval()

    model_clip, _ = clip.load("ViT-L/14@336px", device=device)

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list

    with torch.cuda.amp.autocast(), torch.no_grad():
        if args.dataset == 'mvtec':
            defined_text_prompts = encode_text_with_prompt_ensemble(model_clip, mvtec_class, device, args.dataset, rep_vec='dbscan')
        if args.dataset == 'visa':
            defined_text_prompts = encode_text_with_prompt_ensemble(model_clip, visa_class, device, args.dataset, rep_vec='dbscan')
        if args.dataset == 'mpdd':
            defined_text_prompts = encode_text_with_prompt_ensemble(model_clip, mpdd_class, device, args.dataset, rep_vec='dbscan')
        if args.dataset == 'btad':
            defined_text_prompts = encode_text_with_prompt_ensemble(model_clip, btad_class, device, args.dataset, rep_vec='dbscan')
        if args.dataset == 'sdd':
            defined_text_prompts = encode_text_with_prompt_ensemble(model_clip, sdd_class, device, args.dataset, rep_vec='dbscan')
        if args.dataset == 'dagm':
            defined_text_prompts = encode_text_with_prompt_ensemble(model_clip, dagm_class, device, args.dataset, rep_vec='dbscan')
        if args.dataset == 'dtd':
            defined_text_prompts = encode_text_with_prompt_ensemble(model_clip, dtd_class, device, args.dataset, rep_vec='dbscan')
        if args.dataset == 'headct':
            defined_text_prompts = encode_text_with_prompt_ensemble(model_clip, headct_class, device, args.dataset, rep_vec='dbscan')
        if args.dataset == 'brainmri':
            defined_text_prompts = encode_text_with_prompt_ensemble(model_clip, brainmri_class, device, args.dataset, rep_vec='dbscan')
        if args.dataset == 'br35h':
            defined_text_prompts = encode_text_with_prompt_ensemble(model_clip, br35h_class, device, args.dataset, rep_vec='dbscan')
        if args.dataset == 'isic':
            defined_text_prompts = encode_text_with_prompt_ensemble(model_clip, isic_class, device, args.dataset, rep_vec='dbscan')
        if args.dataset == 'clinicdb':
            defined_text_prompts = encode_text_with_prompt_ensemble(model_clip, clinicdb_class, device, args.dataset, rep_vec='dbscan')
        if args.dataset == 'colondb':
            defined_text_prompts = encode_text_with_prompt_ensemble(model_clip, colondb_class, device, args.dataset, rep_vec='dbscan')
        if args.dataset == 'tn3k':
            defined_text_prompts = encode_text_with_prompt_ensemble(model_clip, tn3k_class, device, args.dataset, rep_vec='dbscan')


    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)
    model.to(device)

    for epoch in tqdm(range(args.epoch)):
        checkpoint = torch.load(os.path.join(args.checkpoint_path, 'epoch_' + str(epoch + 1) + '.pth'))
        print(os.path.join(args.checkpoint_path, 'epoch_' + str(epoch + 1) + '.pth'))
        prompt_learner.load_state_dict(checkpoint["prompt_learner"])
        prompt_learner.to(device)

        results = {}
        metrics = {}
        for obj in obj_list:
            results[obj] = {}
            results[obj]['gt_sp'] = []
            results[obj]['pr_sp'] = []
            results[obj]['imgs_masks'] = []
            results[obj]['anomaly_maps'] = []
            metrics[obj] = {}
            metrics[obj]['pixel-auroc'] = 0
            metrics[obj]['pixel-aupro'] = 0
            metrics[obj]['pixel-f1'] = 0
            metrics[obj]['pixel-ap'] = 0
            metrics[obj]['image-auroc'] = 0
            metrics[obj]['image-f1'] = 0
            metrics[obj]['image-ap'] = 0

        all_text = []
        all_text_defined = []
        all_text_seg = []


        for idx, items in enumerate(tqdm(test_dataloader)):
            image = items['img'].to(device)
            cls_name = items['cls_name']
            cls_id = items['cls_id']
            gt_mask = items['img_mask']
            gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
            results[cls_name[0]]['imgs_masks'].append(gt_mask)  # px
            results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())

            with torch.no_grad():

                # text
                defined_text_features = []
                for cls in cls_name:
                    defined_text_features.append(defined_text_prompts[cls])
                defined_text_features = torch.stack(defined_text_features, dim=0)  # [1, 512, 2]
                defined_text_features = defined_text_features.permute(0, 2, 1)
                # print(defined_text_features.shape)   # 8 768 2

                image_features, patch_features = model.encode_image(image, features_list, DPAM_layer = 20)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
                text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()

                text_features = prompt_learner.ca(text_features, defined_text_features)

                text_features_classify_pos = torch.stack([text_features[0:1, ...]], dim = 0)[0]
                text_features_seg_pos = torch.stack([text_features[1:5, ...]], dim = 0)[0]
                text_features_classify_neg = torch.stack([text_features[5:6, ...]], dim = 0)[0]
                text_features_seg_neg = torch.stack([text_features[6:10, ...]], dim = 0)[0]

                text_features_classify_pos = text_features_classify_pos / text_features_classify_pos.norm(dim=-1, keepdim=True)
                text_features_classify_neg = text_features_classify_neg / text_features_classify_neg.norm(dim=-1, keepdim=True)
                text_features_seg_pos = text_features_seg_pos / text_features_seg_pos.norm(dim=-1, keepdim=True)
                text_features_seg_neg = text_features_seg_neg / text_features_seg_neg.norm(dim=-1, keepdim=True)

                # text_probs = image_features @ text_features_classify.permute(0, 2, 1)
                # text_probs = (text_probs/0.07).softmax(-1)
                # text_probs = text_probs[:, 0, 1]
                anomaly_map_list = []
                for idx, patch_feature in enumerate(patch_features):
                    if idx >= args.feature_map_layer[0]:
                        patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)
                        # text_f = torch.cat((text_features_classify_pos[0, ...].unsqueeze(0), text_features_classify_neg[0, ...,].unsqueeze(0)), dim=0)
                        # text_f = text_f / text_f.norm(dim=-1, keepdim=True)
                        text_f = torch.cat((text_features_seg_pos[idx, ...].unsqueeze(0), text_features_seg_neg[idx, ...,].unsqueeze(0)), dim=0)
                        # text_f = defined_text_features[0].float()
                        similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_f)
                        similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
                        anomaly_map = (similarity_map[...,1] + 1 - similarity_map[...,0])/2.0
                        anomaly_map_list.append(anomaly_map)

                anomaly_map = torch.stack(anomaly_map_list)
            
                text_probs = 0
                for idx in range(len(text_features_classify_pos)):
                    # print(cls.shape)   # 8 1370 768
                #     cls = cls / cls.norm(dim=-1, keepdim=True)
                    text_f = torch.cat((text_features_classify_pos[idx, ...].unsqueeze(0), text_features_classify_neg[idx, ...,].unsqueeze(0)), dim=0)
                    # print(text_f.shape)   # 2 768
                    text_probs_cls = image_features @ text_f.unsqueeze(0).permute(0, 2, 1)
                    text_probs_cls = (text_probs_cls/0.07).softmax(-1)
                    text_probs += text_probs_cls[:, 0, 1]

                text_probs_cls = image_features @ defined_text_features.permute(0, 2, 1).float()
                text_probs_cls = (text_probs_cls/0.07).softmax(-1)
                text_probs += text_probs_cls[:, 0, 1]

                text_probs = (text_probs + anomaly_map.max()) / 2

                anomaly_map = anomaly_map.sum(dim = 0)

                results[cls_name[0]]['pr_sp'].extend(text_probs.detach().cpu())
                anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in anomaly_map.detach().cpu()], dim = 0 )

                results[cls_name[0]]['anomaly_maps'].append(anomaly_map)
                visualizer(items['img_path'], anomaly_map.detach().cpu().numpy(), args.image_size, args.save_path, cls_name)
                # results[cls_name[0]]['anomaly_maps'].append(max_region)
                # visualizer(items['img_path'], max_region.detach().cpu().numpy(), args.image_size, args.save_path, cls_name)



        table_ls = []
        image_auroc_list = []
        image_f1_list = []
        image_ap_list = []
        pixel_auroc_list = []
        pixel_aupro_list = []
        pixel_f1_list = []
        pixel_ap_list = []

        for obj in obj_list:
            table = []
            table.append(obj)
            results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
            results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
            if args.metrics == 'image-level':
                image_auroc = image_level_metrics(results, obj, "image-auroc")
                image_ap = image_level_metrics(results, obj, "image-ap")
                table.append(str(np.round(image_auroc * 100, decimals=1)))
                table.append(str(np.round(image_ap * 100, decimals=1)))
                image_auroc_list.append(image_auroc)
                image_ap_list.append(image_ap) 
            elif args.metrics == 'pixel-level':
                pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
                pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
                table.append(str(np.round(pixel_auroc * 100, decimals=1)))
                table.append(str(np.round(pixel_aupro * 100, decimals=1)))
                pixel_auroc_list.append(pixel_auroc)
                pixel_aupro_list.append(pixel_aupro)
            elif args.metrics == 'image-pixel-level':
                image_auroc = image_level_metrics(results, obj, "image-auroc")
                image_f1 = image_level_metrics(results, obj, "image-f1")
                image_ap = image_level_metrics(results, obj, "image-ap")
                pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
                pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
                pixel_f1 = pixel_level_metrics(results, obj, "pixel-f1")
                pixel_ap = pixel_level_metrics(results, obj, "pixel-ap")
                table.append(str(np.round(pixel_auroc * 100, decimals=1)))
                table.append(str(np.round(pixel_aupro * 100, decimals=1)))
                table.append(str(np.round(pixel_f1 * 100, decimals=1)))
                table.append(str(np.round(pixel_ap * 100, decimals=1)))
                table.append(str(np.round(image_auroc * 100, decimals=1)))
                table.append(str(np.round(image_f1 * 100, decimals=1)))
                table.append(str(np.round(image_ap * 100, decimals=1)))
                image_auroc_list.append(image_auroc)
                image_f1_list.append(image_f1)
                image_ap_list.append(image_ap) 
                pixel_auroc_list.append(pixel_auroc)
                pixel_aupro_list.append(pixel_aupro)
                pixel_f1_list.append(pixel_f1)
                pixel_ap_list.append(pixel_ap)
                """
                image_auroc = image_level_metrics(results, obj, "image-auroc")
                image_ap = image_level_metrics(results, obj, "image-ap")
                pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
                pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
                table.append(str(np.round(pixel_auroc * 100, decimals=1)))
                table.append(str(np.round(pixel_aupro * 100, decimals=1)))
                table.append(str(np.round(image_auroc * 100, decimals=1)))
                table.append(str(np.round(image_ap * 100, decimals=1)))
                image_auroc_list.append(image_auroc)
                image_ap_list.append(image_ap) 
                pixel_auroc_list.append(pixel_auroc)
                pixel_aupro_list.append(pixel_aupro)
                """
            table_ls.append(table)

        if args.metrics == 'image-level':
            # logger
            table_ls.append(['mean', 
                        str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
            results = tabulate(table_ls, headers=['objects', 'image_auroc', 'image_ap'], tablefmt="pipe")
        elif args.metrics == 'pixel-level':
            # logger
            table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1))
                       ])
            results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro'], tablefmt="pipe")
        elif args.metrics == 'image-pixel-level':
            # logger
            table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)), 
                        str(np.round(np.mean(pixel_f1_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_ap_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_f1_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
            results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro', 'pixel_f1', 'pixel_ap', 'image_auroc', 'image_f1', 'image_ap'], tablefmt="pipe")
            """
            table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)), 
                        str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
            results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro', 'image_auroc', 'image_ap'], tablefmt="pipe")
            """
        logger.info("\n%s", results)



if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/', help='path to checkpoint')
    # model
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--epoch", type=int, default=25, help="epochs")

    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int,  nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    test(args)
