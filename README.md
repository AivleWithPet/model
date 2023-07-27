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
├── preprocessing
│   ├── to_coco.py
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

## RTMDet
# Reference
