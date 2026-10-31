"""Early stopping tracks strict improvements, matching the notebook's criterion."""
class EarlyStopping:
    def __init__(self, patience):
        self.patience, self.best, self.wait = patience, -float('inf'), 0

    def update(self, score):
        improved = score > self.best
        if improved:
            self.best, self.wait = score, 0
        else:
            self.wait += 1
        return improved, self.wait >= self.patience
