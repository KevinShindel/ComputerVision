# YOLO + SAM label configuration for Label Studio
# Paste this XML into: Project Settings → Labeling Interface → Code

# ── Combined: polygon (SAM) + bounding box fallback ──────────────────────────
# The ML backend will try to produce PolygonLabels first (from SAM).
# If SAM fails for a detection, it falls back to RectangleLabels.
#
# IMPORTANT: keep name/toName values in sync with .env settings:
#   POLYGON_FROM_NAME = polygon_labels
#   BBOX_FROM_NAME    = bbox_labels
#   IMAGE_TO_NAME     = image
#   IMAGE_VALUE       = image

LABEL_CONFIG = """
<View>
  <Image name="image" value="$image"/>

  <!-- SAM segmentation output (primary) -->
  <PolygonLabels name="polygon_labels" toName="image" strokeWidth="2" opacity="0.5">
    <Label value="person"     background="#FF6B6B"/>
    <Label value="car"        background="#4ECDC4"/>
    <Label value="bicycle"    background="#45B7D1"/>
    <Label value="motorcycle" background="#96CEB4"/>
    <Label value="bus"        background="#FFEAA7"/>
    <Label value="truck"      background="#DDA0DD"/>
    <Label value="dog"        background="#98FB98"/>
    <Label value="cat"        background="#FFB347"/>
  </PolygonLabels>

  <!-- YOLO bbox fallback (shown when SAM mask is unavailable) -->
  <RectangleLabels name="bbox_labels" toName="image" strokeWidth="2">
    <Label value="person"     background="#FF6B6B"/>
    <Label value="car"        background="#4ECDC4"/>
    <Label value="bicycle"    background="#45B7D1"/>
    <Label value="motorcycle" background="#96CEB4"/>
    <Label value="bus"        background="#FFEAA7"/>
    <Label value="truck"      background="#DDA0DD"/>
    <Label value="dog"        background="#98FB98"/>
    <Label value="cat"        background="#FFB347"/>
  </RectangleLabels>
</View>
"""
