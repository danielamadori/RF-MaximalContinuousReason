from typing import Dict, List, Tuple

def icf_to_bitmap_mask(icf: Dict[str, Tuple[float, float]], eu: Dict[str, List[float]]) -> List[int]:
    """
    Convert an ICF (Interval Condition Features) to a bitmap mask based on endpoints universe.

    Args:
        icf: Dictionary mapping feature_name to open interval (inf, sup) where inf < sup
             Intervals can include ±infinity as bounds
        eu: Endpoints universe - dictionary mapping feature_name to monotonic array
            Each monotonic array starts with -inf, ends with +inf, and contains
            unique float values in strictly increasing order

    Returns:
        List[int]: Bitmap mask as concatenation of feature-specific bitmaps
                  in lexicographic order of feature names in eu

    Raises:
        ValueError: If eu keys don't contain all icf keys
        TypeError: If inputs have wrong types
    """
    if not isinstance(icf, dict):
        raise TypeError("icf must be a dictionary")

    if not isinstance(eu, dict):
        raise TypeError("eu must be a dictionary")

    # Check constraint: eu keys must contain icf keys
    icf_keys = set(icf.keys())
    eu_keys = set(eu.keys())

    if not icf_keys.issubset(eu_keys):
        missing_keys = icf_keys - eu_keys
        raise ValueError(f"eu keys must contain all icf keys. Missing keys: {missing_keys}")

    # Process features in lexicographic order
    sorted_features = sorted(eu.keys())

    bitmap_mask = []

    for feature in sorted_features:
        endpoints = eu[feature]

        # Get interval for this feature, or default to (-inf, +inf) if missing in icf
        if feature in icf:
            b, e = icf[feature]  # (inf, sup) interval
        else:
            b, e = float('-inf'), float('inf')

        # Create bitmap for this feature
        feature_bitmap = []

        for endpoint in endpoints:
            # Put 1 if endpoint is in interval [b, e], 0 otherwise
            # Note: interval is open, so b <= endpoint <= e
            if b <= endpoint <= e:
                feature_bitmap.append(1)
            else:
                feature_bitmap.append(0)

        # Add this feature's bitmap to the overall mask
        bitmap_mask.extend(feature_bitmap)

    return bitmap_mask


def bitmap_mask_to_string(bitmap_mask: List[int]) -> str:
    """
    Convert bitmap mask to string representation for easier visualization.

    Args:
        bitmap_mask: List of 0s and 1s

    Returns:
        str: String representation of the bitmap
    """
    return ''.join(map(str, bitmap_mask))


def analyze_bitmap_mask(bitmap_mask: List[int], eu: Dict[str, List[float]]) -> Dict[str, List[int]]:
    """
    Break down bitmap mask by feature for analysis.

    Args:
        bitmap_mask: The complete bitmap mask
        eu: Endpoints universe dictionary

    Returns:
        Dict mapping feature names to their portion of the bitmap mask
    """
    if not isinstance(bitmap_mask, list):
        raise TypeError("bitmap_mask must be a list")

    if not isinstance(eu, dict):
        raise TypeError("eu must be a dictionary")

    sorted_features = sorted(eu.keys())
    feature_bitmaps = {}

    start_idx = 0
    for feature in sorted_features:
        feature_length = len(eu[feature])
        end_idx = start_idx + feature_length

        if end_idx > len(bitmap_mask):
            raise ValueError(f"Bitmap mask too short for feature '{feature}'")

        feature_bitmaps[feature] = bitmap_mask[start_idx:end_idx]
        start_idx = end_idx

    if start_idx != len(bitmap_mask):
        raise ValueError("Bitmap mask length doesn't match total endpoints universe size")

    return feature_bitmaps