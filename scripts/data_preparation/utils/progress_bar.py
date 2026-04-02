from tqdm import tqdm

# 1. Création du Callback personnalisé
class TqdmCallback:
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="LightGBM training")
        
    def __call__(self, env):
        # env.iteration est l'indice actuel de l'arbre
        self.pbar.update(1)
        # Optionnel : afficher la MAE actuelle dans la barre
        if env.evaluation_result_list:
            # On récupère le dernier score (MAE) sur le valid set
            score = env.evaluation_result_list[0][2]
            self.pbar.set_postfix(mae=f"{score:.4f}")