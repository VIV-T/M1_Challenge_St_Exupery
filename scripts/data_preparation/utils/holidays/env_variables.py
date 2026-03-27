START_DATE = "2023-01-01"
END_DATE = "2027-12-31"

FR_ZONES = [
    "Zone A", "Zone B", "Zone C", 
    "Corse", "Guadeloupe", "Réunion", "Martinique", "Guyane", "Mayotte"
]

# --- MAPPING ADMINISTRATIF ---
# Associe un nom d'aéroport à l'entité reconnue par l'Éducation Nationale --- # Hard coded
FR_MAPPING_ACADEMIE = {
    # --- CORSE ---
    "Figari": "Corse",
    "Figari-Sud": "Corse",
    "Ajaccio": "Corse",
    "Bastia": "Corse",
    "Calvi": "Corse",

    # --- OUTRE-MER ---
    "Pointe-a-Pitre": "Guadeloupe",
    "Pointe": "Guadeloupe",
    "St-Denis": "Réunion",
    "Saint-Denis": "Réunion",
    "Fort-de-France": "Martinique",
    "Cayenne": "Guyane",
    "Dzaoudzi": "Mayotte",

    # --- ZONE A (Besançon, Bordeaux, Clermont-Ferrand, Grenoble, Limoges, Lyon, Poitiers, Saint-Étienne) ---
    "Biarritz": "Bordeaux",
    "Pau": "Bordeaux",
    "Chambery": "Grenoble",
    "Annecy": "Grenoble",
    "Saint-Etienne": "Lyon",
    "Clermont": "Clermont-Ferrand",
    "Dole": "Besançon",
    "Pontarlier": "Besançon",
    "La Rochelle": "Poitiers",
    "La-Rochelle": "Poitiers",
    "Larochelle": "Poitiers",
    "Poitiers": "Poitiers",
    "Niort": "Poitiers",
    "Angouleme": "Poitiers",

    # --- ZONE B (Aix-Marseille, Amiens, Caen, Lille, Nancy-Metz, Nantes, Nice, Orléans-Tours, Reims, Rennes, Rouen, Strasbourg) ---
    "Vannes": "Rennes",
    "Brest": "Rennes",
    "Lorient": "Rennes",
    "Quimper": "Rennes",
    "Saint-Jacques": "Rennes",
    "Beauvais": "Amiens",
    "Deauville": "Normandie",
    "Caen": "Normandie",
    "Toulon": "Nice",
    "Marseille": "Aix-Marseille",
    "Metz": "Nancy-Metz",
    "Nancy": "Nancy-Metz",
    "Bale": "Strasbourg",
    "Mulhouse": "Strasbourg",
    "Saint-Louis": "Strasbourg",
    "Brest": "Rennes",

    # --- ZONE C (Créteil, Montpellier, Paris, Toulouse, Versailles) ---
    "Perpignan": "Montpellier",
    "Tarbes": "Toulouse",
    "Rodez": "Toulouse",
    "Carcassonne": "Montpellier",
    "Nimes": "Montpellier",
    "Orly": "Paris",
    "Roissy": "Paris"
}