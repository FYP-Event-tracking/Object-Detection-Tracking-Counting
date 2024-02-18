from ultralytics import YOLO
import supervision as sv
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def initialize_model(model_path):
    model = YOLO(os.path.relpath(model_path))
    model.fuse()
    return model

def process_video(model, source_video_path, target_video_path):
    # Define polygon zone
    polygon = np.array([
        [50, 1000],
        [1050, 1000],
        [1050, 300],
        [50, 300]
    ])

    # Create VideoInfo instance
    video_info = sv.VideoInfo.from_video_path(source_video_path)

    # Create PolygonZone instances
    zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)
    zone_2 = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)

    # Create BoxAnnotator and PolygonZoneAnnotator instances
    box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.white(), thickness=6, text_thickness=6, text_scale=2)
    zone_annotator_2 = sv.PolygonZoneAnnotator(zone=zone_2, color=sv.Color.blue(), thickness=6, text_thickness=6, text_scale=2)

    # Open target video file
    with sv.VideoSink(target_video_path, video_info) as sink:
        in_count = 0
        out_count = 0
        in_check = False
        out_check = False

        # Process each frame in the video
        for result in model.track(source=source_video_path, tracker='bytetrack.yaml', show=False, stream=True, agnostic_nms=True, persist=True):
            frame = result.orig_img
            detections = sv.Detections.from_yolov8(result)

            if result.boxes.id is not None:
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

            labels = [
                f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id
                in detections
            ]

            frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

            detections = detections[detections.class_id == 2]
            zone.trigger(detections=detections)

            if (any(detections.class_id == 2) and any(zone.trigger(detections=detections)) and not in_check):
                in_count += 1
                print(in_count)
                in_check = True
                out_check = False

            detections = sv.Detections.from_yolov8(result)
            detections = detections[detections.class_id == 3]
            zone_2.trigger(detections=detections)

            if (any(detections.class_id == 3) and any(zone_2.trigger(detections=detections)) and not out_check):
                out_count += 1
                print(out_count)
                out_check = True
                in_check = False

            frame = zone_annotator.annotate(scene=frame)
            frame = zone_annotator_2.annotate(scene=frame)

            sink.write_frame(frame)

def main():
    model_path = "best.pt"
    source_video_path = "Videos/video_4_Trim.mp4"
    target_video_path = "Videos/output_video.mp4"

    model = initialize_model(model_path)
    process_video(model, source_video_path, target_video_path)

if __name__ == "__main__":
    main()
