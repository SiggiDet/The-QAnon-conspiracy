import keras_tuner
from model.model_fn import model_fn
from model.training import train_and_evaluate

def tune_hyperparmeters(hp,params,inputs):
    def add_hyper_param(v,name):
        if name == "embedding_size" and "GloVe" in params.embeddings:
            return hp.Choice("embeddings", [50,100,200,300])
        if isinstance(v[0],str):
            return hp.Choice(name, v)
        if isinstance(v[0],bool):
            print(name)
            print(v)
            return hp.Boolean(name)
        if isinstance(v[0],float):
            return hp.Float(name, min_value=v[0], max_value=v[1], sampling="linear")
        if isinstance(v[0],int):
            return hp.Int(name, min_value=v[0], max_value=v[1], sampling="linear")

    for key, value in params:
        if isinstance(value,list):
            params.__dict__[key] = add_hyper_param(value,key)
        #
        # this shit is stupid change it to just put the dicts in a list that way
        #           different optimizers can be passed in
        #
        elif isinstance(value,dict):
            for k, v in value.items():
                if isinstance(v,list):
                    params.__dict__[key][k] = add_hyper_param(v,k)

    train_model, inputs = model_fn(inputs, params)

    return train_model
