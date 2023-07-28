class Trainer():
    def __init__(self,model,data) -> None:
        self.model = model
        self.num_epoch = 1
        self.ds = data
    def train():
        for batch in self.ds:
            output = self.model(batch)
            return output

