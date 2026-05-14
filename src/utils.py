def get_class_names():
    """
    Retourne la liste des noms de classes (sports).
    Ordre alphabétique du dataset Kaggle 100 Sports.
    """
    # Liste des 100 sports (ordre alphabétique)
    SPORTS_LIST = [
        "air hockey", "amateur wrestling", "american football", "archery",
        "arm wrestling", "axe throwing", "balance beam", "barrel racing",
        "baseball", "basketball", "billiards", "bmx", "bobsled", "bowling",
        "boxing", "bull riding", "canoe slalom", "cheerleading", "cliff diving",
        "cricket", "croquet", "curling", "disc golf", "fencing", "field hockey",
        "figure skating pairs", "figure skating women", "fly fishing",
        "football", "formula 1 racing", "frisbee", "gaga ball", "golf",
        "hammer throw", "hang gliding", "harness racing", "high jump",
        "hockey", "horse jumping", "horse racing", "horseshoe pitching",
        "hurdles", "hydroplane racing", "ice climbing", "ice yachting",
        "jai alai", "javelin", "judo", "lacrosse", "luge", "motorcycle racing",
        "mushing", "nascar racing", "olympic wrestling", "parallel bars",
        "pole dancing", "pole vault", "polo", "pommel horse", "rings",
        "rock climbing", "roller derby", "rollerblade racing", "rowing",
        "rugby", "sailboat racing", "shot put", "shuffleboard", "ski jumping",
        "sky surfing", "snowboarding", "snowmobile racing", "speed skating",
        "squash", "sumo wrestling", "surfing", "swimming", "table tennis",
        "tennis", "track bicycle", "trapeze", "tug of war", "ultimate frisbee",
        "uneven bars", "volleyball", "water cycling", "water polo",
        "weightlifting", "wheelchair basketball", "wheelchair racing",
        "wingsuit flying"
    ]
    
    # Vérifie d'abord si un fichier local existe
    classes_path = os.path.join(DATA_DIR, "class_dict.csv")
    if os.path.exists(classes_path):
        import pandas as pd
        df = pd.read_csv(classes_path)
        return sorted(df['class'].unique().tolist())
    
    train_dir = os.path.join(DATA_DIR, "train")
    if os.path.exists(train_dir):
        return sorted(os.listdir(train_dir))
    
    # Fallback : retourne la liste hardcodée
    print("⚠️ Utilisation de la liste hardcodée des sports")
    return SPORTS_LIST