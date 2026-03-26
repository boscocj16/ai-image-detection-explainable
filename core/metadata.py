from PIL import Image
from PIL.ExifTags import TAGS

def extract_metadata(path):
    try:
        img = Image.open(path)
        exif_data = img._getexif()

        if exif_data is None:
            return {}

        metadata = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            metadata[tag] = value

        return metadata

    except Exception:
        return {}
    
def metadata_signal(path):
    metadata = extract_metadata(path)

    if not metadata:
        return 1 

    camera_tags = ["Make", "Model"]

    has_camera_info = any(tag in metadata for tag in camera_tags)

    if not has_camera_info:
        return 1

    return 0