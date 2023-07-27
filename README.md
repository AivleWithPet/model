![header](https://capsule-render.vercel.app/api?type=waving&color=auto&height=300&section=header&text=Model&fontSize=90&animation=fadeIn&fontAlignY=38&desc=프로젝트에%20사용된%20모델에%20관하여%20다룹니다!&descAlignY=51&descAlign=62)

# Table of contents

1. [Overview](#Overview)
2. [Stacks](#Stacks)
3. [Structure](#Structure)
4. [Modeling](#Modeling)

# Overview
반려동물의 안구부위 사진을 이용해 질환을 탐지하는 모델로 MMdetection의 CSPNet&RTMDet모델로 Object Detection Task를 수행하였습니다.

# Stacks
 <div align=center> <img src="https://img.shields.io/badge/python-3776AB?style=flat&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/pytorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white"/> <img src="https://img.shields.io/badge/docker-2496ED?style=flat&logo=Docker&logoColor=white"/> <img src="https://img.shields.io/badge/git-F05032?style=flat&logo=git&logoColor=white"/> <img src="https://img.shields.io/badge/github-181717?style=flat&logo=github&logoColor=white"/> </div>

# Structure

```bash
├── README.md
├── configs
│   └── rtmdet_tiny_8xb32-300e_coco.py
├── docker
│   ├── Dockerfile
├── inference.py
├── mmdet
│   ├── __init__.py
│   ├── apis
│   │   ├── __init__.py
│   │   ├── det_inferencer.py
│   │   └── inference.py
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── api_wrappers
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   └── coco_api.cpython-38.pyc
│   │   │   └── coco_api.py
│   │   ├── base_det_dataset.py
│   │   ├── cityscapes.py
│   │   ├── coco.py
│   │   ├── coco_panoptic.py
│   │   ├── crowdhuman.py
│   │   ├── dataset_wrappers.py
│   │   ├── deepfashion.py
│   │   ├── lvis.py
│   │   ├── objects365.py
│   │   ├── openimages.py
│   │   ├── samplers
│   │   │   ├── __init__.py
│   │   │   ├── batch_sampler.py
│   │   │   ├── class_aware_sampler.py
│   │   │   └── multi_source_sampler.py
│   │   ├── transforms
│   │   │   ├── __init__.py
│   │   │   ├── augment_wrappers.py
│   │   │   ├── colorspace.py
│   │   │   ├── formatting.py
│   │   │   ├── geometric.py
│   │   │   ├── instaboost.py
│   │   │   ├── loading.py
│   │   │   ├── transforms.py
│   │   │   └── wrappers.py
│   │   ├── utils.py
│   │   ├── voc.py
│   │   ├── wider_face.py
│   │   └── xml_style.py
│   ├── engine
│   │   ├── __init__.py
│   │   ├── hooks
│   │   │   ├── __init__.py
│   │   │   ├── checkloss_hook.py
│   │   │   ├── mean_teacher_hook.py
│   │   │   ├── memory_profiler_hook.py
│   │   │   ├── num_class_check_hook.py
│   │   │   ├── pipeline_switch_hook.py
│   │   │   ├── set_epoch_info_hook.py
│   │   │   ├── sync_norm_hook.py
│   │   │   ├── utils.py
│   │   │   ├── visualization_hook.py
│   │   │   └── yolox_mode_switch_hook.py
│   │   ├── optimizers
│   │   │   ├── __init__.py
│   │   │   └── layer_decay_optimizer_constructor.py
│   │   ├── runner
│   │   │   ├── __init__.py
│   │   │   └── loops.py
│   │   └── schedulers
│   │       ├── __init__.py
│   │       └── quadratic_warmup.py
│   ├── evaluation
│   │   ├── __init__.py
│   │   ├── functional
│   │   │   ├── __init__.py
│   │   │   ├── bbox_overlaps.py
│   │   │   ├── cityscapes_utils.py
│   │   │   ├── class_names.py
│   │   │   ├── mean_ap.py
│   │   │   ├── panoptic_utils.py
│   │   │   └── recall.py
│   │   └── metrics
│   │       ├── __init__.py
│   │       ├── cityscapes_metric.py
│   │       ├── coco_metric.py
│   │       ├── coco_occluded_metric.py
│   │       ├── coco_panoptic_metric.py
│   │       ├── crowdhuman_metric.py
│   │       ├── dump_det_results.py
│   │       ├── dump_proposals_metric.py
│   │       ├── lvis_metric.py
│   │       ├── openimages_metric.py
│   │       └── voc_metric.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   └── __init__.cpython-38.pyc
│   │   ├── backbones
│   │   │   ├── __init__.py
│   │   │   ├── csp_darknet.py
│   │   │   ├── cspnext.py
│   │   │   ├── darknet.py
│   │   │   ├── detectors_resnet.py
│   │   │   ├── detectors_resnext.py
│   │   │   ├── efficientnet.py
│   │   │   ├── hourglass.py
│   │   │   ├── hrnet.py
│   │   │   ├── mobilenet_v2.py
│   │   │   ├── pvt.py
│   │   │   ├── regnet.py
│   │   │   ├── res2net.py
│   │   │   ├── resnest.py
│   │   │   ├── resnet.py
│   │   │   ├── resnext.py
│   │   │   ├── ssd_vgg.py
│   │   │   ├── swin.py
│   │   │   └── trident_resnet.py
│   │   ├── data_preprocessors
│   │   │   ├── __init__.py
│   │   │   └── data_preprocessor.py
│   │   ├── dense_heads
│   │   │   ├── __init__.py
│   │   │   ├── anchor_free_head.py
│   │   │   ├── anchor_head.py
│   │   │   ├── atss_head.py
│   │   │   ├── autoassign_head.py
│   │   │   ├── base_dense_head.py
│   │   │   ├── base_mask_head.py
│   │   │   ├── boxinst_head.py
│   │   │   ├── cascade_rpn_head.py
│   │   │   ├── centernet_head.py
│   │   │   ├── centernet_update_head.py
│   │   │   ├── centripetal_head.py
│   │   │   ├── condinst_head.py
│   │   │   ├── conditional_detr_head.py
│   │   │   ├── corner_head.py
│   │   │   ├── dab_detr_head.py
│   │   │   ├── ddod_head.py
│   │   │   ├── deformable_detr_head.py
│   │   │   ├── dense_test_mixins.py
│   │   │   ├── detr_head.py
│   │   │   ├── dino_head.py
│   │   │   ├── embedding_rpn_head.py
│   │   │   ├── fcos_head.py
│   │   │   ├── fovea_head.py
│   │   │   ├── free_anchor_retina_head.py
│   │   │   ├── fsaf_head.py
│   │   │   ├── ga_retina_head.py
│   │   │   ├── ga_rpn_head.py
│   │   │   ├── gfl_head.py
│   │   │   ├── guided_anchor_head.py
│   │   │   ├── lad_head.py
│   │   │   ├── ld_head.py
│   │   │   ├── mask2former_head.py
│   │   │   ├── maskformer_head.py
│   │   │   ├── nasfcos_head.py
│   │   │   ├── paa_head.py
│   │   │   ├── pisa_retinanet_head.py
│   │   │   ├── pisa_ssd_head.py
│   │   │   ├── reppoints_head.py
│   │   │   ├── retina_head.py
│   │   │   ├── retina_sepbn_head.py
│   │   │   ├── rpn_head.py
│   │   │   ├── rtmdet_head.py
│   │   │   ├── rtmdet_ins_head.py
│   │   │   ├── sabl_retina_head.py
│   │   │   ├── solo_head.py
│   │   │   ├── solov2_head.py
│   │   │   ├── ssd_head.py
│   │   │   ├── tood_head.py
│   │   │   ├── vfnet_head.py
│   │   │   ├── yolact_head.py
│   │   │   ├── yolo_head.py
│   │   │   ├── yolof_head.py
│   │   │   └── yolox_head.py
│   │   ├── detectors
│   │   │   ├── __init__.py
│   │   │   ├── atss.py
│   │   │   ├── autoassign.py
│   │   │   ├── base.py
│   │   │   ├── base_detr.py
│   │   │   ├── boxinst.py
│   │   │   ├── cascade_rcnn.py
│   │   │   ├── centernet.py
│   │   │   ├── condinst.py
│   │   │   ├── conditional_detr.py
│   │   │   ├── cornernet.py
│   │   │   ├── crowddet.py
│   │   │   ├── d2_wrapper.py
│   │   │   ├── dab_detr.py
│   │   │   ├── ddod.py
│   │   │   ├── deformable_detr.py
│   │   │   ├── detr.py
│   │   │   ├── dino.py
│   │   │   ├── fast_rcnn.py
│   │   │   ├── faster_rcnn.py
│   │   │   ├── fcos.py
│   │   │   ├── fovea.py
│   │   │   ├── fsaf.py
│   │   │   ├── gfl.py
│   │   │   ├── grid_rcnn.py
│   │   │   ├── htc.py
│   │   │   ├── kd_one_stage.py
│   │   │   ├── lad.py
│   │   │   ├── mask2former.py
│   │   │   ├── mask_rcnn.py
│   │   │   ├── mask_scoring_rcnn.py
│   │   │   ├── maskformer.py
│   │   │   ├── nasfcos.py
│   │   │   ├── paa.py
│   │   │   ├── panoptic_fpn.py
│   │   │   ├── panoptic_two_stage_segmentor.py
│   │   │   ├── point_rend.py
│   │   │   ├── queryinst.py
│   │   │   ├── reppoints_detector.py
│   │   │   ├── retinanet.py
│   │   │   ├── rpn.py
│   │   │   ├── rtmdet.py
│   │   │   ├── scnet.py
│   │   │   ├── semi_base.py
│   │   │   ├── single_stage.py
│   │   │   ├── single_stage_instance_seg.py
│   │   │   ├── soft_teacher.py
│   │   │   ├── solo.py
│   │   │   ├── solov2.py
│   │   │   ├── sparse_rcnn.py
│   │   │   ├── tood.py
│   │   │   ├── trident_faster_rcnn.py
│   │   │   ├── two_stage.py
│   │   │   ├── vfnet.py
│   │   │   ├── yolact.py
│   │   │   ├── yolo.py
│   │   │   ├── yolof.py
│   │   │   └── yolox.py
│   │   ├── layers
│   │   │   ├── __init__.py
│   │   │   ├── activations.py
│   │   │   ├── bbox_nms.py
│   │   │   ├── brick_wrappers.py
│   │   │   ├── conv_upsample.py
│   │   │   ├── csp_layer.py
│   │   │   ├── dropblock.py
│   │   │   ├── ema.py
│   │   │   ├── inverted_residual.py
│   │   │   ├── matrix_nms.py
│   │   │   ├── msdeformattn_pixel_decoder.py
│   │   │   ├── normed_predictor.py
│   │   │   ├── pixel_decoder.py
│   │   │   ├── positional_encoding.py
│   │   │   ├── res_layer.py
│   │   │   ├── se_layer.py
│   │   │   └── transformer
│   │   │       ├── __init__.py
│   │   │       ├── conditional_detr_layers.py
│   │   │       ├── dab_detr_layers.py
│   │   │       ├── deformable_detr_layers.py
│   │   │       ├── detr_layers.py
│   │   │       ├── dino_layers.py
│   │   │       ├── mask2former_layers.py
│   │   │       └── utils.py
│   │   ├── losses
│   │   │   ├── __init__.py
│   │   │   ├── accuracy.py
│   │   │   ├── ae_loss.py
│   │   │   ├── balanced_l1_loss.py
│   │   │   ├── cross_entropy_loss.py
│   │   │   ├── dice_loss.py
│   │   │   ├── focal_loss.py
│   │   │   ├── gaussian_focal_loss.py
│   │   │   ├── gfocal_loss.py
│   │   │   ├── ghm_loss.py
│   │   │   ├── iou_loss.py
│   │   │   ├── kd_loss.py
│   │   │   ├── mse_loss.py
│   │   │   ├── pisa_loss.py
│   │   │   ├── seesaw_loss.py
│   │   │   ├── smooth_l1_loss.py
│   │   │   ├── utils.py
│   │   │   └── varifocal_loss.py
│   │   ├── necks
│   │   │   ├── __init__.py
│   │   │   ├── bfp.py
│   │   │   ├── channel_mapper.py
│   │   │   ├── cspnext_pafpn.py
│   │   │   ├── ct_resnet_neck.py
│   │   │   ├── dilated_encoder.py
│   │   │   ├── dyhead.py
│   │   │   ├── fpg.py
│   │   │   ├── fpn.py
│   │   │   ├── fpn_carafe.py
│   │   │   ├── hrfpn.py
│   │   │   ├── nas_fpn.py
│   │   │   ├── nasfcos_fpn.py
│   │   │   ├── pafpn.py
│   │   │   ├── rfp.py
│   │   │   ├── ssd_neck.py
│   │   │   ├── ssh.py
│   │   │   ├── yolo_neck.py
│   │   │   └── yolox_pafpn.py
│   │   ├── roi_heads
│   │   │   ├── __init__.py
│   │   │   ├── base_roi_head.py
│   │   │   ├── bbox_heads
│   │   │   │   ├── __init__.py
│   │   │   │   ├── __pycache__
│   │   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   │   ├── bbox_head.cpython-38.pyc
│   │   │   │   │   ├── convfc_bbox_head.cpython-38.pyc
│   │   │   │   │   ├── dii_head.cpython-38.pyc
│   │   │   │   │   ├── double_bbox_head.cpython-38.pyc
│   │   │   │   │   ├── multi_instance_bbox_head.cpython-38.pyc
│   │   │   │   │   ├── sabl_head.cpython-38.pyc
│   │   │   │   │   └── scnet_bbox_head.cpython-38.pyc
│   │   │   │   ├── bbox_head.py
│   │   │   │   ├── convfc_bbox_head.py
│   │   │   │   ├── dii_head.py
│   │   │   │   ├── double_bbox_head.py
│   │   │   │   ├── multi_instance_bbox_head.py
│   │   │   │   ├── sabl_head.py
│   │   │   │   └── scnet_bbox_head.py
│   │   │   ├── cascade_roi_head.py
│   │   │   ├── double_roi_head.py
│   │   │   ├── dynamic_roi_head.py
│   │   │   ├── grid_roi_head.py
│   │   │   ├── htc_roi_head.py
│   │   │   ├── mask_heads
│   │   │   │   ├── __init__.py
│   │   │   │   ├── coarse_mask_head.py
│   │   │   │   ├── dynamic_mask_head.py
│   │   │   │   ├── fcn_mask_head.py
│   │   │   │   ├── feature_relay_head.py
│   │   │   │   ├── fused_semantic_head.py
│   │   │   │   ├── global_context_head.py
│   │   │   │   ├── grid_head.py
│   │   │   │   ├── htc_mask_head.py
│   │   │   │   ├── mask_point_head.py
│   │   │   │   ├── maskiou_head.py
│   │   │   │   ├── scnet_mask_head.py
│   │   │   │   └── scnet_semantic_head.py
│   │   │   ├── mask_scoring_roi_head.py
│   │   │   ├── multi_instance_roi_head.py
│   │   │   ├── pisa_roi_head.py
│   │   │   ├── point_rend_roi_head.py
│   │   │   ├── roi_extractors
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base_roi_extractor.py
│   │   │   │   ├── generic_roi_extractor.py
│   │   │   │   └── single_level_roi_extractor.py
│   │   │   ├── scnet_roi_head.py
│   │   │   ├── shared_heads
│   │   │   │   ├── __init__.py
│   │   │   │   └── res_layer.py
│   │   │   ├── sparse_roi_head.py
│   │   │   ├── standard_roi_head.py
│   │   │   ├── test_mixins.py
│   │   │   └── trident_roi_head.py
│   │   ├── seg_heads
│   │   │   ├── __init__.py
│   │   │   ├── base_semantic_head.py
│   │   │   ├── panoptic_fpn_head.py
│   │   │   └── panoptic_fusion_heads
│   │   │       ├── __init__.py
│   │   │       ├── base_panoptic_fusion_head.py
│   │   │       ├── heuristic_fusion_head.py
│   │   │       └── maskformer_fusion_head.py
│   │   ├── task_modules
│   │   │   ├── __init__.py
│   │   │   ├── assigners
│   │   │   │   ├── __init__.py
│   │   │   │   ├── approx_max_iou_assigner.py
│   │   │   │   ├── assign_result.py
│   │   │   │   ├── atss_assigner.py
│   │   │   │   ├── base_assigner.py
│   │   │   │   ├── center_region_assigner.py
│   │   │   │   ├── dynamic_soft_label_assigner.py
│   │   │   │   ├── grid_assigner.py
│   │   │   │   ├── hungarian_assigner.py
│   │   │   │   ├── iou2d_calculator.py
│   │   │   │   ├── match_cost.py
│   │   │   │   ├── max_iou_assigner.py
│   │   │   │   ├── multi_instance_assigner.py
│   │   │   │   ├── point_assigner.py
│   │   │   │   ├── region_assigner.py
│   │   │   │   ├── sim_ota_assigner.py
│   │   │   │   ├── task_aligned_assigner.py
│   │   │   │   └── uniform_assigner.py
│   │   │   ├── builder.py
│   │   │   ├── coders
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base_bbox_coder.py
│   │   │   │   ├── bucketing_bbox_coder.py
│   │   │   │   ├── delta_xywh_bbox_coder.py
│   │   │   │   ├── distance_point_bbox_coder.py
│   │   │   │   ├── legacy_delta_xywh_bbox_coder.py
│   │   │   │   ├── pseudo_bbox_coder.py
│   │   │   │   ├── tblr_bbox_coder.py
│   │   │   │   └── yolo_bbox_coder.py
│   │   │   ├── prior_generators
│   │   │   │   ├── __init__.py
│   │   │   │   ├── anchor_generator.py
│   │   │   │   ├── point_generator.py
│   │   │   │   └── utils.py
│   │   │   └── samplers
│   │   │       ├── __init__.py
│   │   │       ├── base_sampler.py
│   │   │       ├── combined_sampler.py
│   │   │       ├── instance_balanced_pos_sampler.py
│   │   │       ├── iou_balanced_neg_sampler.py
│   │   │       ├── mask_pseudo_sampler.py
│   │   │       ├── mask_sampling_result.py
│   │   │       ├── multi_instance_random_sampler.py
│   │   │       ├── multi_instance_sampling_result.py
│   │   │       ├── ohem_sampler.py
│   │   │       ├── pseudo_sampler.py
│   │   │       ├── random_sampler.py
│   │   │       ├── sampling_result.py
│   │   │       └── score_hlr_sampler.py
│   │   ├── test_time_augs
│   │   │   ├── __init__.py
│   │   │   ├── det_tta.py
│   │   │   └── merge_augs.py
│   │   └── utils
│   │       ├── __init__.py
│   │       ├── gaussian_target.py
│   │       ├── make_divisible.py
│   │       ├── misc.py
│   │       ├── panoptic_gt_processing.py
│   │       └── point_sample.py
│   ├── registry.py
│   ├── structures
│   │   ├── __init__.py
│   │   ├── bbox
│   │   │   ├── __init__.py
│   │   │   ├── base_boxes.py
│   │   │   ├── bbox_overlaps.py
│   │   │   ├── box_type.py
│   │   │   ├── horizontal_boxes.py
│   │   │   └── transforms.py
│   │   ├── det_data_sample.py
│   │   └── mask
│   │       ├── __init__.py
│   │       ├── mask_target.py
│   │       ├── structures.py
│   │       └── utils.py
│   ├── testing
│   │   ├── __init__.py
│   │   ├── _fast_stop_training_hook.py
│   │   └── _utils.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── benchmark.py
│   │   ├── collect_env.py
│   │   ├── compat_config.py
│   │   ├── contextmanagers.py
│   │   ├── dist_utils.py
│   │   ├── logger.py
│   │   ├── memory.py
│   │   ├── misc.py
│   │   ├── profiling.py
│   │   ├── replace_cfg_vals.py
│   │   ├── setup_env.py
│   │   ├── split_batch.py
│   │   ├── typing_utils.py
│   │   ├── util_mixins.py
│   │   └── util_random.py
│   ├── version.py
│   └── visualization
│       ├── __init__.py
│       ├── local_visualizer.py
│       └── palette.py
├── model-index.yml
├── pytest.ini
├── requirements
│   ├── albu.txt
│   ├── build.txt
│   ├── docs.txt
│   ├── mminstall.txt
│   ├── optional.txt
│   ├── readthedocs.txt
│   ├── runtime.txt
│   └── tests.txt
├── requirements.txt
├── setup.cfg
└── setup.py

```

# Modeling
## CSPNet
### Cross Stage Partial Network

---

![Untitled](https://user-images.githubusercontent.com/101624956/256540476-9b599685-248f-4cd4-9c05-dccf882ab710.png)

그림 2의 a는 DenseNet의 구조, 그림 2의 b는 one stage CSPDenseNet의 구조

DenseNet은 각 stage에서 dense block, transition layer를 포함하는데, 각 dense block은 k개의 dense layer로 구성 

i번째 dense layer의 출력은 i번째 dense layer의 입력과 concat하여 i + 1번째 dense layer의 입력으로 날려줌

이를 도식화하면 아래와 같음

이 때 * 은 convolution operation, w와 x는 각각 weights와 output

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5ac95a0a-9547-4499-9332-137c2a41c925/Untitled.png)

여기서 backpropagation을 사용해 weights를 update하면 식은 아래와 같음

이 때 f는 weight updating function, g는 dense layer의 propagated gradient

직관적으로 보기에도 각각의 다른 dense layer를 update 하면서 똑같은 **gradient가 재사용**되는 것을 확인할 수 있음

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8c349dc2-b545-46a4-92c7-6a7ed7cdb718/Untitled.png)

CSPDenseNet의 하나의 stage는 Partial dense block, Partial transition layer로 구성

Partial dense block은 base layer의 **feature map을** channel을 통해 **두 가지로 나눔**

$$
⁍ = ⁍
$$

여기서 전자인 $x_0’$은 직접적으로 **stage의 끝 부분에 연결**, 후자인 $x_0’’$은 **dense block을 통과**하게 된다.

Dense layer의 출력 값($x_0’’$$, x_1, x_2, …$)은 transition layer를 거쳐 $x_t$가 되고, $x_t$는 아까 나눠진 $x_0’$와 concatenation되어 다음 transition layer를 통과해 $x_u$를 생성한다.

이를 도식화 하면 아래와 같다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/daf770cd-5ad2-46cf-9020-b59b422667b5/Untitled.png)

이처럼 분할된 gradient의 flow를 두 path로 나누어 dense block을 통과하지 않는 $x_0’$의 gradient는 복사되지 않고, stage의 마지막 부분에만 추가된다. 이를 통해 과도한 양의 gradient information 복사를 방지한다.

## Partial Dense Block

---

Partial Dense Block은 3가지 목적으로 설계됨.

1. gradient path 증가
    
    feature map을 분할 및 병합하면서 pathgradient path를 2배로 만들고, 이를 통해 위에서 언급한 duplicated gradient 문제를 완화함
    
2. layer 간의 연산량 균형
    
    DenseNet은 layer간의 input과 output을 concatenation하면서 channel 수가 급격하게 커짐.
    
    Partial Dense Block은 base layer의 feature map이 반으로 분할되므로 dense layer에서 사용되는 channel 수 또한 감소하게 됨
    
3. memory traffic 감소
    
    base layer의 feature map이 반으로 분할되면서 연산량도 반으로 감소
    

아래 그림은 여러 종류의 fusion 방법을 나타냄

## Partial Transition Layer

---

Partial Transition Layer의 목적인 gradient 조합의 차이를 최대화 하는 것.

gradient flow를 split, concatenation, split, concatenation하면서 layer의 중복된 gradient 학습을 완화하고 duplication의 가능성을 감소시킨다.

## Fusion Strategy

---

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1152b813-2ec5-433f-ac30-f777c71c935a/Untitled.png)

1. a는 기존 single path를 가진 DenseNet
2. b는 CSPDenseNet으로 분할된 feature 중 $x_0’’$이 Dense Block을 거치고 Transition layer를 통과한 후, $x_0’$과 concatenation되어 한 번 더 Transition layer를 통과함
3. c는 Dense layer를 통과한 $x_0’’$이 Transition layer를 통과하지 않고, $x_0’$과 concatenation 후 Transition layer를 통과한다.
4. d는 Dense layer를 통과한 $x_0’’$이 Transition layer를 통과하고 $x_0’$과 concatenation하지만, 이후 Transition layer를 통과하지 않는다.

아래 결과에 따르면 b 전략이 가장 기존보다 효과가 탁월했음을 알 수 있음.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e94e7531-fa8f-45bb-ae2d-72c8b9360860/Untitled.png)

## Exact Fusion Model(EFM)

---

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1f542116-6ae7-48ae-a101-3acf6fb661df/Untitled.png)

feature pyramid를 생성하는 과정을 제안함.

YOLOv3의 FPN은 backbone을 거치면서 나온 각 layer의 feature map을 각각 prediction하고 output으로 날림

ThunderNet의 GFM은 backbone을 거치면서 나온 각 layer의 feature을 하나로 합친 후, 이를 나누어 prediction, output으로 날림

여기서 제안한 EFM은 위의 두 가지를 적절히 섞은 느낌으로 backbone을 거치면서 나온 feature map을 적절히 조합하면서 prediction하고 output으로 날림

아래는 제안한 EFM의 성능

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9206e095-bc23-49c9-8e12-841cb8344ad2/Untitled.png)

# Reference

---

[CSPNet: A New Backbone that can Enhance Learning Capability of CNN](https://arxiv.org/abs/1911.11929)

[RTMDet: An Empirical Study of Designing Real-Time Object Detectors](https://arxiv.org/abs/2212.07784)

[mmdetection/configs/rtmdet at main · open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet)

## RTMDet
