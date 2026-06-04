import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble_task_multiscale_seg_prompt import AnomalyCLIP_PromptLearner
from prompt_ensemble_clipad import encode_text_with_prompt_ensemble

from loss import FocalLoss, BinaryDiceLoss
from utils import normalize
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
from utils import get_transform
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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):

    logger = get_logger(args.save_path)

    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details = AnomalyCLIP_parameters)
    model.eval()

    model_clip, _ = clip.load("ViT-L/14@336px", device=device)

    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    with torch.cuda.amp.autocast(), torch.no_grad():
        if args.dataset == 'mvtec':
            defined_text_prompts = encode_text_with_prompt_ensemble(model_clip, mvtec_class, device, args.dataset, rep_vec='dbscan')
        if args.dataset == 'visa':
            defined_text_prompts = encode_text_with_prompt_ensemble(model_clip, visa_class, device, args.dataset, rep_vec='dbscan')

  ##########################################################################################
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)
    ##########################################################################################
    optimizer = torch.optim.Adam(list(prompt_learner.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    
    
    model.eval()
    prompt_learner.train()
    for epoch in tqdm(range(args.epoch)):
        model.eval()
        prompt_learner.train()
        loss_list = []
        image_loss_list = []

        for items in tqdm(train_dataloader):
            image = items['img'].to(device)
            label = items['anomaly']
            cls_name = items['cls_name']

            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            with torch.no_grad():
                # Apply DPAM to the layer from 6 to 24
                # DPAM_layer represents the number of layer refined by DPAM from top to bottom
                # DPAM_layer = 1, no DPAM is used
                # DPAM_layer = 20 as default
                image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer = 20)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
            trainable_num = sum(p.numel() for p in prompt_learner.parameters() if p.requires_grad)
            print(trainable_num)
           ####################################
            # text
            defined_text_features = []
            for cls in cls_name:
                defined_text_features.append(defined_text_prompts[cls])
            defined_text_features = torch.stack(defined_text_features, dim=0)  # [1, 512, 2]
            defined_text_features = defined_text_features.permute(0, 2, 1)
            # print(defined_text_features.shape)   # 8 768 2

            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()

            text_features_classify_pos = torch.stack([text_features[0:1, ...]], dim = 0)[0]
            text_features_seg_pos = torch.stack([text_features[1:5, ...]], dim = 0)[0]
            text_features_classify_neg = torch.stack([text_features[5:6, ...]], dim = 0)[0]
            text_features_seg_neg = torch.stack([text_features[6:10, ...]], dim = 0)[0]

            # text_features_classify = torch.stack(torch.chunk(text_features_classify, dim = 0, chunks = 2), dim = 1)
            # text_features_seg = torch.stack(torch.chunk(text_features_seg, dim = 0, chunks = 2), dim = 1)

            text_features_classify_pos = text_features_classify_pos / text_features_classify_pos.norm(dim=-1, keepdim=True)
            text_features_classify_neg = text_features_classify_neg / text_features_classify_neg.norm(dim=-1, keepdim=True)
            text_features_seg_pos = text_features_seg_pos / text_features_seg_pos.norm(dim=-1, keepdim=True)
            text_features_seg_neg = text_features_seg_neg / text_features_seg_neg.norm(dim=-1, keepdim=True)

            # Apply DPAM surgery
            # text_probs = image_features.unsqueeze(1) @ text_features_classify.permute(0, 2, 1)
            # text_probs = text_probs[:, 0, ...]/0.07
            # print(text_probs.shape)   # 1 2
            # image_loss = F.cross_entropy(text_probs.squeeze(), label.long().cuda())
            # image_loss = F.cross_entropy(text_probs, label.long().cuda())
            # image_loss_list.append(image_loss.item())
            #########################################################################
            similarity_map_list = []
            # similarity_map_list.append(similarity_map)
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)
                    text_f = torch.cat((text_features_seg_pos[idx, ...].unsqueeze(0), text_features_seg_neg[idx, ...,].unsqueeze(0)), dim=0)
                    # print(text_f.shape)   # 2 768
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_f)
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)
                    similarity_map_list.append(similarity_map)

            loss = 0
            for i in range(len(similarity_map_list)):
                loss += loss_focal(similarity_map_list[i], gt)
                loss += loss_dice(similarity_map_list[i][:, 1, :, :], gt)
                loss += loss_dice(similarity_map_list[i][:, 0, :, :], 1-gt)

            image_loss = 0
            for idx in range(len(text_features_classify_pos)):
                # print(cls.shape)   # 8 1370 768
                # cls = cls / cls.norm(dim=-1, keepdim=True)
                text_f = torch.cat((text_features_classify_pos[idx, ...].unsqueeze(0), text_features_classify_neg[idx, ...,].unsqueeze(0)), dim=0)
                # print(text_f.shape)   # 2 768
                # print(cls[:, :1, :].shape)   # 8 1 768
                text_probs_cls = (image_features @ (text_f.unsqueeze(0).permute(0, 2, 1))).permute(1, 0, 2)
                # print(text_probs_cls.shape)   # 8 1 2
                text_probs_cls = text_probs_cls[:, 0, ...]/0.07
                # print(text_probs_cls.shape)  # 8 2
                image_loss += F.cross_entropy(text_probs_cls, label.long().cuda())

            image_loss_list.append(image_loss.item())

            optimizer.zero_grad()
            (loss+image_loss).backward()
            optimizer.step()
            loss_list.append(loss.item())

        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}, image_loss:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_list), np.mean(image_loss_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({"prompt_learner": prompt_learner.state_dict()}, ckp_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./data/visa", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoint', help='path to save results')


    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")

    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")

    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
