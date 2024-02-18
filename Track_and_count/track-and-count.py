from ultralytics import YOLO
import supervision as sv
import os
import numpy as np


# Initialize YOLO model
model_path = "best.pt"
model = YOLO(os.path.relpath(model_path))
model.fuse()

# Define paths
SOURCE_VIDEO_PATH = "Videos/video_4_Trim.mp4"
TARGET_VIDEO_PATH = "Videos/output_video.mp4"

# Define global variables
list_of_dict = []
in_count = 0
out_count = 0
in_check = False
out_check = False

# Additional variables
polygon = np.array([
    [50, 1000],
    [1050, 1000],
    [1050, 300],
    [50, 300]
])

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
generator = sv.video.get_video_frames_generator(SOURCE_VIDEO_PATH)
zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)
zone_2 = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)
box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.white(), thickness=6, text_thickness=6, text_scale=2)
zone_annotator_2 = sv.PolygonZoneAnnotator(zone=zone_2, color=sv.Color.blue(), thickness=6, text_thickness=6, text_scale=2)

# Main function
def main():
    with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
        for result in model.track(source=SOURCE_VIDEO_PATH, tracker='bytetrack.yaml', show=False, stream=True, agnostic_nms=True, persist=True ):
            process_frame(result, sink)

# Additional function to process each frame
def process_frame(result, sink):
    global in_count, out_count, in_check, out_check
    
    frame = result.orig_img
    detections = sv.Detections.from_yolov8(result)

    if result.boxes.id is not None:
        detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

    labels = [
        f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id in detections
    ]

    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

    detections = detections[detections.class_id == 2]
    zone.trigger(detections=detections)

    if (any(detections.class_id == 2) and  any(zone.trigger(detections=detections)) and in_check == False):
        in_count = in_count + 1
        print(in_count)
        in_check = True
        out_check = False

    detections = sv.Detections.from_yolov8(result)
    detections = detections[detections.class_id == 3]
    zone_2.trigger(detections=detections)

    if (any(detections.class_id == 3) and  any(zone_2.trigger(detections=detections)) and out_check == False):
        out_count = out_count + 1
        print(out_count)
        out_check = True
        in_check = False

    frame = zone_annotator.annotate(scene=frame)
    frame = zone_annotator_2.annotate(scene=frame)

    sink.write_frame(frame)

# Call the main function
if __name__ == "__main__":
    main()
