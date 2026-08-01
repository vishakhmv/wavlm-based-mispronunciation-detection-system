
PHONOLOGICAL_FEATURES = [
    "consonant", "sonorant", "fricative", "nasal", "stop", "approximant", "affricate", 
    "liquid", "vowel", "semivowel", "continuant",
    "alveolar", "palatal", "dental", "glottal", "labial", "velar", "mid", "high", "low", 
    "front", "back", "central", "anterior", "posterior", "retroflex", "bilabial", "coronal", "dorsal",
    "long", "short", "monophthong", "diphthong", "round", "voiced"
]

PHONEME_TO_FEATURES = {
    # Consonants
    "P":  ["consonant", "stop", "labial", "bilabial", "anterior"],
    "B":  ["consonant", "stop", "labial", "bilabial", "anterior", "voiced"],
    "T":  ["consonant", "stop", "alveolar", "coronal", "anterior"],
    "D":  ["consonant", "stop", "alveolar", "coronal", "anterior", "voiced"],
    "K":  ["consonant", "stop", "velar", "dorsal", "posterior"],
    "G":  ["consonant", "stop", "velar", "dorsal", "posterior", "voiced"],
    "CH": ["consonant", "affricate", "palatal", "coronal", "posterior"],
    "JH": ["consonant", "affricate", "palatal", "coronal", "posterior", "voiced"],
    "F":  ["consonant", "fricative", "labial", "anterior", "continuant"],
    "V":  ["consonant", "fricative", "labial", "anterior", "continuant", "voiced"],
    "TH": ["consonant", "fricative", "dental", "coronal", "anterior", "continuant"],
    "DH": ["consonant", "fricative", "dental", "coronal", "anterior", "continuant", "voiced"],
    "S":  ["consonant", "fricative", "alveolar", "coronal", "anterior", "continuant"],
    "Z":  ["consonant", "fricative", "alveolar", "coronal", "anterior", "continuant", "voiced"],
    "SH": ["consonant", "fricative", "palatal", "coronal", "posterior", "continuant"],
    "ZH": ["consonant", "fricative", "palatal", "coronal", "posterior", "continuant", "voiced"],
    "HH": ["consonant", "fricative", "glottal", "continuant"],
    "M":  ["consonant", "sonorant", "nasal", "labial", "bilabial", "anterior", "voiced"],
    "N":  ["consonant", "sonorant", "nasal", "alveolar", "coronal", "anterior", "voiced"],
    "NG": ["consonant", "sonorant", "nasal", "velar", "dorsal", "posterior", "voiced"],
    "L":  ["consonant", "sonorant", "liquid", "alveolar", "coronal", "anterior", "continuant", "voiced"],
    "R":  ["consonant", "sonorant", "liquid", "palatal", "coronal", "posterior", "retroflex", "continuant", "voiced"],
    "Y":  ["consonant", "sonorant", "semivowel", "palatal", "dorsal", "continuant", "voiced"],
    "W":  ["consonant", "sonorant", "semivowel", "labial", "velar", "dorsal", "round", "continuant", "voiced"],

    # Vowels
    "AA": ["vowel", "sonorant", "low", "back", "monophthong", "voiced", "long"],
    "AE": ["vowel", "sonorant", "low", "front", "monophthong", "voiced", "short"],
    "AH": ["vowel", "sonorant", "mid", "central", "monophthong", "voiced", "short"],
    "AO": ["vowel", "sonorant", "mid", "back", "round", "monophthong", "voiced", "long"],
    "AW": ["vowel", "sonorant", "diphthong", "back", "round", "voiced", "long"],
    "AY": ["vowel", "sonorant", "diphthong", "central", "front", "voiced", "long"],
    "EH": ["vowel", "sonorant", "mid", "front", "monophthong", "voiced", "short"],
    "ER": ["vowel", "sonorant", "mid", "central", "retroflex", "monophthong", "voiced", "long"],
    "EY": ["vowel", "sonorant", "diphthong", "front", "voiced", "long"],
    "IH": ["vowel", "sonorant", "high", "front", "monophthong", "voiced", "short"],
    "IY": ["vowel", "sonorant", "high", "front", "monophthong", "voiced", "long"],
    "OW": ["vowel", "sonorant", "diphthong", "back", "round", "voiced", "long"],
    "OY": ["vowel", "sonorant", "diphthong", "back", "front", "round", "voiced", "long"],
    "UH": ["vowel", "sonorant", "high", "back", "round", "monophthong", "voiced", "short"],
    "UW": ["vowel", "sonorant", "high", "back", "round", "monophthong", "voiced", "long"]
}


TIMIT_61_TO_39 = {
    'aa': 'AA', 'ae': 'AE', 'ah': 'AH', 'ao': 'AO', 'aw': 'AW', 'ay': 'AY',
    'b': 'B', 'ch': 'CH', 'd': 'D', 'dh': 'DH', 'eh': 'EH', 'er': 'ER', 'ey': 'EY',
    'f': 'F', 'g': 'G', 'hh': 'HH', 'ih': 'IH', 'iy': 'IY', 'jh': 'JH',
    'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'ng': 'NG', 'ow': 'OW', 'oy': 'OY',
    'p': 'P', 'r': 'R', 's': 'S', 'sh': 'SH', 't': 'T', 'th': 'TH', 'uh': 'UH',
    'uw': 'UW', 'v': 'V', 'w': 'W', 'y': 'Y', 'z': 'Z', 'zh': 'ZH',
    'ux': 'UW', 'ax': 'AH', 'ix': 'IH', 'axr': 'ER', 'ax-h': 'AH', 
    'em': 'M', 'en': 'N', 'eng': 'NG', 'nx': 'N', 'el': 'L',
    'dx': 'D', 'hv': 'HH',
    'pcl': 'P', 'tcl': 'T', 'kcl': 'K', 'bcl': 'B', 'dcl': 'D', 'gcl': 'G',
    'epi': 'sil', 'pau': 'sil', 'h#': 'sil', 'q': 'sil'
}

import logging

logger = logging.getLogger(__name__)

def _get_features_list(phoneme: str) -> list:
    """Helper to safely fetch features and warn on unknown phonemes."""
    # Clean up L2-ARCTIC transcription artifacts
    phoneme = phoneme.upper().strip("0123_`()")
    
    # Map common alternative phonemes
    if phoneme == 'AX':
        phoneme = 'AH'
    elif phoneme == 'SPN':
        phoneme = 'SIL'
        
    features = PHONEME_TO_FEATURES.get(phoneme)
    if features is None:
        if phoneme.lower() not in ['sil', '|', '']:
            logger.warning(f"Unknown phoneme encountered: '{phoneme}'")
        return []
    return features

def get_feature_vector(phoneme: str) -> list:
    """Returns a binary list of length 35 representing the features of a phoneme."""
    features = _get_features_list(phoneme)
    return [1 if f in features else 0 for f in PHONOLOGICAL_FEATURES]

def phoneme_to_features(phonemes: list) -> dict:
    """Converts a list of phonemes into a dictionary of feature sequences."""
    features_dict = {feat: [] for feat in PHONOLOGICAL_FEATURES}
    for phoneme in phonemes:
        features = _get_features_list(phoneme)
        for feat in PHONOLOGICAL_FEATURES:
            features_dict[feat].append(1 if feat in features else 0)
    return features_dict
