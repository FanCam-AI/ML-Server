import math
import torch
from types import SimpleNamespace
import os
from tracking.OSTrack.lib.test.tracker.ostrack import OSTrack
from tracking.OSTrack.lib.config.ostrack.config import cfg, update_config_from_file

class OSTrackTracker:
    def __init__(self):
        # ---------------------------------
        # 1. config
        # ---------------------------------
        update_config_from_file(
            os.path.join(
                "tracking/OSTrack",
                "experiments/ostrack/vitb_256_mae_ce_32x4_ep300.yaml"
            ),
            cfg
        )

        # ---------------------------------
        # 2. params 생성
        # ---------------------------------
        params = SimpleNamespace()
        params.cfg = cfg
        params.checkpoint = os.path.join(
            "tracking",
            "saved_models",
            "vitb_256_mae_ce_32x4_ep300.pth"
        )
        params.debug = False
        params.save_all_boxes = False

        params.template_factor = cfg.TEST.TEMPLATE_FACTOR
        params.template_size = cfg.TEST.TEMPLATE_SIZE
        params.search_factor = cfg.TEST.SEARCH_FACTOR
        params.search_size = cfg.TEST.SEARCH_SIZE

        # ---------------------------------
        # 3. tracker 생성
        # ---------------------------------
        self.tracker = OSTrack(params, dataset_name="demo")
        self.prev_box = None

        if torch.cuda.is_available():
            self.tracker.network.cuda()
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.initialized = False

    def init(self, frame, bbox):
        """
        bbox: (x, y, w, h)
        """
        self.tracker.initialize(frame, {
            'init_bbox': list(bbox)
        })
        self.initialized = True
        return True

    def update(self, frame, distance_threshold=50, skip_distance_check=False):
        if not self.initialized:
            return False, None

        outputs = self.tracker.track(frame)
        curr_bbox = outputs['target_bbox']  # [x, y, w, h]
        curr_bbox = tuple(map(float, curr_bbox))

        # 🔹 첫 update 호출
        if self.prev_box is None:
            self.prev_box = curr_bbox
            return True, curr_bbox

        if skip_distance_check:
            self.prev_box = curr_bbox
            return True, curr_bbox

        # 중심점 계산
        prev_cx, prev_cy = self.bbox_center(self.prev_box)
        curr_cx, curr_cy = self.bbox_center(curr_bbox)

        # 중심점 거리
        distance = math.hypot(curr_cx - prev_cx, curr_cy - prev_cy)

        # bbox 선택
        if distance > distance_threshold:
            final_bbox = self.prev_box
        else:
            final_bbox = curr_bbox

        # prev_box 갱신
        self.prev_box = final_bbox

        return True, final_bbox

    @staticmethod
    def bbox_center(bbox):
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        return cx, cy