from nn4n.layer import LinearLayer

if __name__ == "__main__":
    input_dim = 100
    output_dim = 1000
    
    linear_layer = LinearLayer(
        input_dim=input_dim,
        output_dim=output_dim,
        weight="uniform",
        bias="uniform",
    )