![header](https://capsule-render.vercel.app/api?type=waving&color=auto&height=300&section=header&text=Model&fontSize=90&animation=fadeIn&fontAlignY=38&desc=프로젝트에%20사용된%20모델에%20관하여%20다룹니다!&descAlignY=51&descAlign=62)

# Table of contents

1. [Overview](#Overview)
2. [Stacks](#Stacks)
3. [Structure](#Structure)

# Overview
반려동물의 안구부위 사진을 이용해 질환을 탐지하는 모델로 MMdetection을 이용해 Object Detection Task를 수행하였습니다.

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
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   ├── det_inferencer.cpython-38.pyc
│   │   │   └── inference.cpython-38.pyc
│   │   ├── det_inferencer.py
│   │   └── inference.py
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   ├── base_det_dataset.cpython-38.pyc
│   │   │   ├── cityscapes.cpython-38.pyc
│   │   │   ├── coco.cpython-38.pyc
│   │   │   ├── coco_panoptic.cpython-38.pyc
│   │   │   ├── crowdhuman.cpython-38.pyc
│   │   │   ├── dataset_wrappers.cpython-38.pyc
│   │   │   ├── deepfashion.cpython-38.pyc
│   │   │   ├── lvis.cpython-38.pyc
│   │   │   ├── objects365.cpython-38.pyc
│   │   │   ├── openimages.cpython-38.pyc
│   │   │   ├── utils.cpython-38.pyc
│   │   │   ├── voc.cpython-38.pyc
│   │   │   ├── wider_face.cpython-38.pyc
│   │   │   └── xml_style.cpython-38.pyc
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
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   ├── batch_sampler.cpython-38.pyc
│   │   │   │   ├── class_aware_sampler.cpython-38.pyc
│   │   │   │   └── multi_source_sampler.cpython-38.pyc
│   │   │   ├── batch_sampler.py
│   │   │   ├── class_aware_sampler.py
│   │   │   └── multi_source_sampler.py
│   │   ├── transforms
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   ├── augment_wrappers.cpython-38.pyc
│   │   │   │   ├── colorspace.cpython-38.pyc
│   │   │   │   ├── formatting.cpython-38.pyc
│   │   │   │   ├── geometric.cpython-38.pyc
│   │   │   │   ├── instaboost.cpython-38.pyc
│   │   │   │   ├── loading.cpython-38.pyc
│   │   │   │   ├── transforms.cpython-38.pyc
│   │   │   │   └── wrappers.cpython-38.pyc
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
│   │   ├── __pycache__
│   │   │   └── __init__.cpython-38.pyc
│   │   ├── hooks
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   ├── checkloss_hook.cpython-38.pyc
│   │   │   │   ├── mean_teacher_hook.cpython-38.pyc
│   │   │   │   ├── memory_profiler_hook.cpython-38.pyc
│   │   │   │   ├── num_class_check_hook.cpython-38.pyc
│   │   │   │   ├── pipeline_switch_hook.cpython-38.pyc
│   │   │   │   ├── set_epoch_info_hook.cpython-38.pyc
│   │   │   │   ├── sync_norm_hook.cpython-38.pyc
│   │   │   │   ├── utils.cpython-38.pyc
│   │   │   │   ├── visualization_hook.cpython-38.pyc
│   │   │   │   └── yolox_mode_switch_hook.cpython-38.pyc
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
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   └── layer_decay_optimizer_constructor.cpython-38.pyc
│   │   │   └── layer_decay_optimizer_constructor.py
│   │   ├── runner
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   └── loops.cpython-38.pyc
│   │   │   └── loops.py
│   │   └── schedulers
│   │       ├── __init__.py
│   │       ├── __pycache__
│   │       │   ├── __init__.cpython-38.pyc
│   │       │   └── quadratic_warmup.cpython-38.pyc
│   │       └── quadratic_warmup.py
│   ├── evaluation
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   └── __init__.cpython-38.pyc
│   │   ├── functional
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   ├── bbox_overlaps.cpython-38.pyc
│   │   │   │   ├── cityscapes_utils.cpython-38.pyc
│   │   │   │   ├── class_names.cpython-38.pyc
│   │   │   │   ├── mean_ap.cpython-38.pyc
│   │   │   │   ├── panoptic_utils.cpython-38.pyc
│   │   │   │   └── recall.cpython-38.pyc
│   │   │   ├── bbox_overlaps.py
│   │   │   ├── cityscapes_utils.py
│   │   │   ├── class_names.py
│   │   │   ├── mean_ap.py
│   │   │   ├── panoptic_utils.py
│   │   │   └── recall.py
│   │   └── metrics
│   │       ├── __init__.py
│   │       ├── __pycache__
│   │       │   ├── __init__.cpython-38.pyc
│   │       │   ├── cityscapes_metric.cpython-38.pyc
│   │       │   ├── coco_metric.cpython-38.pyc
│   │       │   ├── coco_occluded_metric.cpython-38.pyc
│   │       │   ├── coco_panoptic_metric.cpython-38.pyc
│   │       │   ├── crowdhuman_metric.cpython-38.pyc
│   │       │   ├── dump_det_results.cpython-38.pyc
│   │       │   ├── dump_proposals_metric.cpython-38.pyc
│   │       │   ├── lvis_metric.cpython-38.pyc
│   │       │   ├── openimages_metric.cpython-38.pyc
│   │       │   └── voc_metric.cpython-38.pyc
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
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   ├── csp_darknet.cpython-38.pyc
│   │   │   │   ├── cspnext.cpython-38.pyc
│   │   │   │   ├── darknet.cpython-38.pyc
│   │   │   │   ├── detectors_resnet.cpython-38.pyc
│   │   │   │   ├── detectors_resnext.cpython-38.pyc
│   │   │   │   ├── efficientnet.cpython-38.pyc
│   │   │   │   ├── hourglass.cpython-38.pyc
│   │   │   │   ├── hrnet.cpython-38.pyc
│   │   │   │   ├── mobilenet_v2.cpython-38.pyc
│   │   │   │   ├── pvt.cpython-38.pyc
│   │   │   │   ├── regnet.cpython-38.pyc
│   │   │   │   ├── res2net.cpython-38.pyc
│   │   │   │   ├── resnest.cpython-38.pyc
│   │   │   │   ├── resnet.cpython-38.pyc
│   │   │   │   ├── resnext.cpython-38.pyc
│   │   │   │   ├── ssd_vgg.cpython-38.pyc
│   │   │   │   ├── swin.cpython-38.pyc
│   │   │   │   └── trident_resnet.cpython-38.pyc
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
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   └── data_preprocessor.cpython-38.pyc
│   │   │   └── data_preprocessor.py
│   │   ├── dense_heads
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   ├── anchor_free_head.cpython-38.pyc
│   │   │   │   ├── anchor_head.cpython-38.pyc
│   │   │   │   ├── atss_head.cpython-38.pyc
│   │   │   │   ├── autoassign_head.cpython-38.pyc
│   │   │   │   ├── base_dense_head.cpython-38.pyc
│   │   │   │   ├── base_mask_head.cpython-38.pyc
│   │   │   │   ├── boxinst_head.cpython-38.pyc
│   │   │   │   ├── cascade_rpn_head.cpython-38.pyc
│   │   │   │   ├── centernet_head.cpython-38.pyc
│   │   │   │   ├── centernet_update_head.cpython-38.pyc
│   │   │   │   ├── centripetal_head.cpython-38.pyc
│   │   │   │   ├── condinst_head.cpython-38.pyc
│   │   │   │   ├── conditional_detr_head.cpython-38.pyc
│   │   │   │   ├── corner_head.cpython-38.pyc
│   │   │   │   ├── dab_detr_head.cpython-38.pyc
│   │   │   │   ├── ddod_head.cpython-38.pyc
│   │   │   │   ├── deformable_detr_head.cpython-38.pyc
│   │   │   │   ├── detr_head.cpython-38.pyc
│   │   │   │   ├── dino_head.cpython-38.pyc
│   │   │   │   ├── embedding_rpn_head.cpython-38.pyc
│   │   │   │   ├── fcos_head.cpython-38.pyc
│   │   │   │   ├── fovea_head.cpython-38.pyc
│   │   │   │   ├── free_anchor_retina_head.cpython-38.pyc
│   │   │   │   ├── fsaf_head.cpython-38.pyc
│   │   │   │   ├── ga_retina_head.cpython-38.pyc
│   │   │   │   ├── ga_rpn_head.cpython-38.pyc
│   │   │   │   ├── gfl_head.cpython-38.pyc
│   │   │   │   ├── guided_anchor_head.cpython-38.pyc
│   │   │   │   ├── lad_head.cpython-38.pyc
│   │   │   │   ├── ld_head.cpython-38.pyc
│   │   │   │   ├── mask2former_head.cpython-38.pyc
│   │   │   │   ├── maskformer_head.cpython-38.pyc
│   │   │   │   ├── nasfcos_head.cpython-38.pyc
│   │   │   │   ├── paa_head.cpython-38.pyc
│   │   │   │   ├── pisa_retinanet_head.cpython-38.pyc
│   │   │   │   ├── pisa_ssd_head.cpython-38.pyc
│   │   │   │   ├── reppoints_head.cpython-38.pyc
│   │   │   │   ├── retina_head.cpython-38.pyc
│   │   │   │   ├── retina_sepbn_head.cpython-38.pyc
│   │   │   │   ├── rpn_head.cpython-38.pyc
│   │   │   │   ├── rtmdet_head.cpython-38.pyc
│   │   │   │   ├── rtmdet_ins_head.cpython-38.pyc
│   │   │   │   ├── sabl_retina_head.cpython-38.pyc
│   │   │   │   ├── solo_head.cpython-38.pyc
│   │   │   │   ├── solov2_head.cpython-38.pyc
│   │   │   │   ├── ssd_head.cpython-38.pyc
│   │   │   │   ├── tood_head.cpython-38.pyc
│   │   │   │   ├── vfnet_head.cpython-38.pyc
│   │   │   │   ├── yolact_head.cpython-38.pyc
│   │   │   │   ├── yolo_head.cpython-38.pyc
│   │   │   │   ├── yolof_head.cpython-38.pyc
│   │   │   │   └── yolox_head.cpython-38.pyc
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
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   ├── atss.cpython-38.pyc
│   │   │   │   ├── autoassign.cpython-38.pyc
│   │   │   │   ├── base.cpython-38.pyc
│   │   │   │   ├── base_detr.cpython-38.pyc
│   │   │   │   ├── boxinst.cpython-38.pyc
│   │   │   │   ├── cascade_rcnn.cpython-38.pyc
│   │   │   │   ├── centernet.cpython-38.pyc
│   │   │   │   ├── condinst.cpython-38.pyc
│   │   │   │   ├── conditional_detr.cpython-38.pyc
│   │   │   │   ├── cornernet.cpython-38.pyc
│   │   │   │   ├── crowddet.cpython-38.pyc
│   │   │   │   ├── d2_wrapper.cpython-38.pyc
│   │   │   │   ├── dab_detr.cpython-38.pyc
│   │   │   │   ├── ddod.cpython-38.pyc
│   │   │   │   ├── deformable_detr.cpython-38.pyc
│   │   │   │   ├── detr.cpython-38.pyc
│   │   │   │   ├── dino.cpython-38.pyc
│   │   │   │   ├── fast_rcnn.cpython-38.pyc
│   │   │   │   ├── faster_rcnn.cpython-38.pyc
│   │   │   │   ├── fcos.cpython-38.pyc
│   │   │   │   ├── fovea.cpython-38.pyc
│   │   │   │   ├── fsaf.cpython-38.pyc
│   │   │   │   ├── gfl.cpython-38.pyc
│   │   │   │   ├── grid_rcnn.cpython-38.pyc
│   │   │   │   ├── htc.cpython-38.pyc
│   │   │   │   ├── kd_one_stage.cpython-38.pyc
│   │   │   │   ├── lad.cpython-38.pyc
│   │   │   │   ├── mask2former.cpython-38.pyc
│   │   │   │   ├── mask_rcnn.cpython-38.pyc
│   │   │   │   ├── mask_scoring_rcnn.cpython-38.pyc
│   │   │   │   ├── maskformer.cpython-38.pyc
│   │   │   │   ├── nasfcos.cpython-38.pyc
│   │   │   │   ├── paa.cpython-38.pyc
│   │   │   │   ├── panoptic_fpn.cpython-38.pyc
│   │   │   │   ├── panoptic_two_stage_segmentor.cpython-38.pyc
│   │   │   │   ├── point_rend.cpython-38.pyc
│   │   │   │   ├── queryinst.cpython-38.pyc
│   │   │   │   ├── reppoints_detector.cpython-38.pyc
│   │   │   │   ├── retinanet.cpython-38.pyc
│   │   │   │   ├── rpn.cpython-38.pyc
│   │   │   │   ├── rtmdet.cpython-38.pyc
│   │   │   │   ├── scnet.cpython-38.pyc
│   │   │   │   ├── semi_base.cpython-38.pyc
│   │   │   │   ├── single_stage.cpython-38.pyc
│   │   │   │   ├── single_stage_instance_seg.cpython-38.pyc
│   │   │   │   ├── soft_teacher.cpython-38.pyc
│   │   │   │   ├── solo.cpython-38.pyc
│   │   │   │   ├── solov2.cpython-38.pyc
│   │   │   │   ├── sparse_rcnn.cpython-38.pyc
│   │   │   │   ├── tood.cpython-38.pyc
│   │   │   │   ├── trident_faster_rcnn.cpython-38.pyc
│   │   │   │   ├── two_stage.cpython-38.pyc
│   │   │   │   ├── vfnet.cpython-38.pyc
│   │   │   │   ├── yolact.cpython-38.pyc
│   │   │   │   ├── yolo.cpython-38.pyc
│   │   │   │   ├── yolof.cpython-38.pyc
│   │   │   │   └── yolox.cpython-38.pyc
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
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   ├── activations.cpython-38.pyc
│   │   │   │   ├── bbox_nms.cpython-38.pyc
│   │   │   │   ├── brick_wrappers.cpython-38.pyc
│   │   │   │   ├── conv_upsample.cpython-38.pyc
│   │   │   │   ├── csp_layer.cpython-38.pyc
│   │   │   │   ├── dropblock.cpython-38.pyc
│   │   │   │   ├── ema.cpython-38.pyc
│   │   │   │   ├── inverted_residual.cpython-38.pyc
│   │   │   │   ├── matrix_nms.cpython-38.pyc
│   │   │   │   ├── msdeformattn_pixel_decoder.cpython-38.pyc
│   │   │   │   ├── normed_predictor.cpython-38.pyc
│   │   │   │   ├── pixel_decoder.cpython-38.pyc
│   │   │   │   ├── positional_encoding.cpython-38.pyc
│   │   │   │   ├── res_layer.cpython-38.pyc
│   │   │   │   └── se_layer.cpython-38.pyc
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
│   │   │       ├── __pycache__
│   │   │       │   ├── __init__.cpython-38.pyc
│   │   │       │   ├── conditional_detr_layers.cpython-38.pyc
│   │   │       │   ├── dab_detr_layers.cpython-38.pyc
│   │   │       │   ├── deformable_detr_layers.cpython-38.pyc
│   │   │       │   ├── detr_layers.cpython-38.pyc
│   │   │       │   ├── dino_layers.cpython-38.pyc
│   │   │       │   ├── mask2former_layers.cpython-38.pyc
│   │   │       │   └── utils.cpython-38.pyc
│   │   │       ├── conditional_detr_layers.py
│   │   │       ├── dab_detr_layers.py
│   │   │       ├── deformable_detr_layers.py
│   │   │       ├── detr_layers.py
│   │   │       ├── dino_layers.py
│   │   │       ├── mask2former_layers.py
│   │   │       └── utils.py
│   │   ├── losses
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   ├── accuracy.cpython-38.pyc
│   │   │   │   ├── ae_loss.cpython-38.pyc
│   │   │   │   ├── balanced_l1_loss.cpython-38.pyc
│   │   │   │   ├── cross_entropy_loss.cpython-38.pyc
│   │   │   │   ├── dice_loss.cpython-38.pyc
│   │   │   │   ├── focal_loss.cpython-38.pyc
│   │   │   │   ├── gaussian_focal_loss.cpython-38.pyc
│   │   │   │   ├── gfocal_loss.cpython-38.pyc
│   │   │   │   ├── ghm_loss.cpython-38.pyc
│   │   │   │   ├── iou_loss.cpython-38.pyc
│   │   │   │   ├── kd_loss.cpython-38.pyc
│   │   │   │   ├── mse_loss.cpython-38.pyc
│   │   │   │   ├── pisa_loss.cpython-38.pyc
│   │   │   │   ├── seesaw_loss.cpython-38.pyc
│   │   │   │   ├── smooth_l1_loss.cpython-38.pyc
│   │   │   │   ├── utils.cpython-38.pyc
│   │   │   │   └── varifocal_loss.cpython-38.pyc
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
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   ├── bfp.cpython-38.pyc
│   │   │   │   ├── channel_mapper.cpython-38.pyc
│   │   │   │   ├── cspnext_pafpn.cpython-38.pyc
│   │   │   │   ├── ct_resnet_neck.cpython-38.pyc
│   │   │   │   ├── dilated_encoder.cpython-38.pyc
│   │   │   │   ├── dyhead.cpython-38.pyc
│   │   │   │   ├── fpg.cpython-38.pyc
│   │   │   │   ├── fpn.cpython-38.pyc
│   │   │   │   ├── fpn_carafe.cpython-38.pyc
│   │   │   │   ├── hrfpn.cpython-38.pyc
│   │   │   │   ├── nas_fpn.cpython-38.pyc
│   │   │   │   ├── nasfcos_fpn.cpython-38.pyc
│   │   │   │   ├── pafpn.cpython-38.pyc
│   │   │   │   ├── rfp.cpython-38.pyc
│   │   │   │   ├── ssd_neck.cpython-38.pyc
│   │   │   │   ├── ssh.cpython-38.pyc
│   │   │   │   ├── yolo_neck.cpython-38.pyc
│   │   │   │   └── yolox_pafpn.cpython-38.pyc
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
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   ├── base_roi_head.cpython-38.pyc
│   │   │   │   ├── cascade_roi_head.cpython-38.pyc
│   │   │   │   ├── double_roi_head.cpython-38.pyc
│   │   │   │   ├── dynamic_roi_head.cpython-38.pyc
│   │   │   │   ├── grid_roi_head.cpython-38.pyc
│   │   │   │   ├── htc_roi_head.cpython-38.pyc
│   │   │   │   ├── mask_scoring_roi_head.cpython-38.pyc
│   │   │   │   ├── multi_instance_roi_head.cpython-38.pyc
│   │   │   │   ├── pisa_roi_head.cpython-38.pyc
│   │   │   │   ├── point_rend_roi_head.cpython-38.pyc
│   │   │   │   ├── scnet_roi_head.cpython-38.pyc
│   │   │   │   ├── sparse_roi_head.cpython-38.pyc
│   │   │   │   ├── standard_roi_head.cpython-38.pyc
│   │   │   │   └── trident_roi_head.cpython-38.pyc
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
│   │   │   │   ├── __pycache__
│   │   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   │   ├── coarse_mask_head.cpython-38.pyc
│   │   │   │   │   ├── dynamic_mask_head.cpython-38.pyc
│   │   │   │   │   ├── fcn_mask_head.cpython-38.pyc
│   │   │   │   │   ├── feature_relay_head.cpython-38.pyc
│   │   │   │   │   ├── fused_semantic_head.cpython-38.pyc
│   │   │   │   │   ├── global_context_head.cpython-38.pyc
│   │   │   │   │   ├── grid_head.cpython-38.pyc
│   │   │   │   │   ├── htc_mask_head.cpython-38.pyc
│   │   │   │   │   ├── mask_point_head.cpython-38.pyc
│   │   │   │   │   ├── maskiou_head.cpython-38.pyc
│   │   │   │   │   ├── scnet_mask_head.cpython-38.pyc
│   │   │   │   │   └── scnet_semantic_head.cpython-38.pyc
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
│   │   │   │   ├── __pycache__
│   │   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   │   ├── base_roi_extractor.cpython-38.pyc
│   │   │   │   │   ├── generic_roi_extractor.cpython-38.pyc
│   │   │   │   │   └── single_level_roi_extractor.cpython-38.pyc
│   │   │   │   ├── base_roi_extractor.py
│   │   │   │   ├── generic_roi_extractor.py
│   │   │   │   └── single_level_roi_extractor.py
│   │   │   ├── scnet_roi_head.py
│   │   │   ├── shared_heads
│   │   │   │   ├── __init__.py
│   │   │   │   ├── __pycache__
│   │   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   │   └── res_layer.cpython-38.pyc
│   │   │   │   └── res_layer.py
│   │   │   ├── sparse_roi_head.py
│   │   │   ├── standard_roi_head.py
│   │   │   ├── test_mixins.py
│   │   │   └── trident_roi_head.py
│   │   ├── seg_heads
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   ├── base_semantic_head.cpython-38.pyc
│   │   │   │   └── panoptic_fpn_head.cpython-38.pyc
│   │   │   ├── base_semantic_head.py
│   │   │   ├── panoptic_fpn_head.py
│   │   │   └── panoptic_fusion_heads
│   │   │       ├── __init__.py
│   │   │       ├── __pycache__
│   │   │       │   ├── __init__.cpython-38.pyc
│   │   │       │   ├── base_panoptic_fusion_head.cpython-38.pyc
│   │   │       │   ├── heuristic_fusion_head.cpython-38.pyc
│   │   │       │   └── maskformer_fusion_head.cpython-38.pyc
│   │   │       ├── base_panoptic_fusion_head.py
│   │   │       ├── heuristic_fusion_head.py
│   │   │       └── maskformer_fusion_head.py
│   │   ├── task_modules
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   └── builder.cpython-38.pyc
│   │   │   ├── assigners
│   │   │   │   ├── __init__.py
│   │   │   │   ├── __pycache__
│   │   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   │   ├── approx_max_iou_assigner.cpython-38.pyc
│   │   │   │   │   ├── assign_result.cpython-38.pyc
│   │   │   │   │   ├── atss_assigner.cpython-38.pyc
│   │   │   │   │   ├── base_assigner.cpython-38.pyc
│   │   │   │   │   ├── center_region_assigner.cpython-38.pyc
│   │   │   │   │   ├── dynamic_soft_label_assigner.cpython-38.pyc
│   │   │   │   │   ├── grid_assigner.cpython-38.pyc
│   │   │   │   │   ├── hungarian_assigner.cpython-38.pyc
│   │   │   │   │   ├── iou2d_calculator.cpython-38.pyc
│   │   │   │   │   ├── match_cost.cpython-38.pyc
│   │   │   │   │   ├── max_iou_assigner.cpython-38.pyc
│   │   │   │   │   ├── multi_instance_assigner.cpython-38.pyc
│   │   │   │   │   ├── point_assigner.cpython-38.pyc
│   │   │   │   │   ├── region_assigner.cpython-38.pyc
│   │   │   │   │   ├── sim_ota_assigner.cpython-38.pyc
│   │   │   │   │   ├── task_aligned_assigner.cpython-38.pyc
│   │   │   │   │   └── uniform_assigner.cpython-38.pyc
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
│   │   │   │   ├── __pycache__
│   │   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   │   ├── base_bbox_coder.cpython-38.pyc
│   │   │   │   │   ├── bucketing_bbox_coder.cpython-38.pyc
│   │   │   │   │   ├── delta_xywh_bbox_coder.cpython-38.pyc
│   │   │   │   │   ├── distance_point_bbox_coder.cpython-38.pyc
│   │   │   │   │   ├── legacy_delta_xywh_bbox_coder.cpython-38.pyc
│   │   │   │   │   ├── pseudo_bbox_coder.cpython-38.pyc
│   │   │   │   │   ├── tblr_bbox_coder.cpython-38.pyc
│   │   │   │   │   └── yolo_bbox_coder.cpython-38.pyc
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
│   │   │   │   ├── __pycache__
│   │   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   │   ├── anchor_generator.cpython-38.pyc
│   │   │   │   │   ├── point_generator.cpython-38.pyc
│   │   │   │   │   └── utils.cpython-38.pyc
│   │   │   │   ├── anchor_generator.py
│   │   │   │   ├── point_generator.py
│   │   │   │   └── utils.py
│   │   │   └── samplers
│   │   │       ├── __init__.py
│   │   │       ├── __pycache__
│   │   │       │   ├── __init__.cpython-38.pyc
│   │   │       │   ├── base_sampler.cpython-38.pyc
│   │   │       │   ├── combined_sampler.cpython-38.pyc
│   │   │       │   ├── instance_balanced_pos_sampler.cpython-38.pyc
│   │   │       │   ├── iou_balanced_neg_sampler.cpython-38.pyc
│   │   │       │   ├── mask_pseudo_sampler.cpython-38.pyc
│   │   │       │   ├── mask_sampling_result.cpython-38.pyc
│   │   │       │   ├── multi_instance_random_sampler.cpython-38.pyc
│   │   │       │   ├── multi_instance_sampling_result.cpython-38.pyc
│   │   │       │   ├── ohem_sampler.cpython-38.pyc
│   │   │       │   ├── pseudo_sampler.cpython-38.pyc
│   │   │       │   ├── random_sampler.cpython-38.pyc
│   │   │       │   ├── sampling_result.cpython-38.pyc
│   │   │       │   └── score_hlr_sampler.cpython-38.pyc
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
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   ├── det_tta.cpython-38.pyc
│   │   │   │   └── merge_augs.cpython-38.pyc
│   │   │   ├── det_tta.py
│   │   │   └── merge_augs.py
│   │   └── utils
│   │       ├── __init__.py
│   │       ├── __pycache__
│   │       │   ├── __init__.cpython-38.pyc
│   │       │   ├── gaussian_target.cpython-38.pyc
│   │       │   ├── make_divisible.cpython-38.pyc
│   │       │   ├── misc.cpython-38.pyc
│   │       │   ├── panoptic_gt_processing.cpython-38.pyc
│   │       │   └── point_sample.cpython-38.pyc
│   │       ├── gaussian_target.py
│   │       ├── make_divisible.py
│   │       ├── misc.py
│   │       ├── panoptic_gt_processing.py
│   │       └── point_sample.py
│   ├── registry.py
│   ├── structures
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   └── det_data_sample.cpython-38.pyc
│   │   ├── bbox
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   ├── base_boxes.cpython-38.pyc
│   │   │   │   ├── bbox_overlaps.cpython-38.pyc
│   │   │   │   ├── box_type.cpython-38.pyc
│   │   │   │   ├── horizontal_boxes.cpython-38.pyc
│   │   │   │   └── transforms.cpython-38.pyc
│   │   │   ├── base_boxes.py
│   │   │   ├── bbox_overlaps.py
│   │   │   ├── box_type.py
│   │   │   ├── horizontal_boxes.py
│   │   │   └── transforms.py
│   │   ├── det_data_sample.py
│   │   └── mask
│   │       ├── __init__.py
│   │       ├── __pycache__
│   │       │   ├── __init__.cpython-38.pyc
│   │       │   ├── mask_target.cpython-38.pyc
│   │       │   ├── structures.cpython-38.pyc
│   │       │   └── utils.cpython-38.pyc
│   │       ├── mask_target.py
│   │       ├── structures.py
│   │       └── utils.py
│   ├── testing
│   │   ├── __init__.py
│   │   ├── _fast_stop_training_hook.py
│   │   └── _utils.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   ├── collect_env.cpython-38.pyc
│   │   │   ├── compat_config.cpython-38.pyc
│   │   │   ├── dist_utils.cpython-38.pyc
│   │   │   ├── logger.cpython-38.pyc
│   │   │   ├── memory.cpython-38.pyc
│   │   │   ├── misc.cpython-38.pyc
│   │   │   ├── replace_cfg_vals.cpython-38.pyc
│   │   │   ├── setup_env.cpython-38.pyc
│   │   │   ├── split_batch.cpython-38.pyc
│   │   │   ├── typing_utils.cpython-38.pyc
│   │   │   ├── util_mixins.cpython-38.pyc
│   │   │   └── util_random.cpython-38.pyc
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
│       ├── __pycache__
│       │   ├── __init__.cpython-38.pyc
│       │   ├── local_visualizer.cpython-38.pyc
│       │   └── palette.cpython-38.pyc
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

# 
