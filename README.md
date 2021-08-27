# pokemon DCGAN

image generator which generate pokemon images!


## usage
```python
from train_pokemon_DCGAN import generate_pokemon
generate_pokemon()
```
the output image will be saved in the `new_predictions` folder


## training:
first run the `process_data()` function
then call the `train()` function. 
```python
from train_pokemon_DCGAN import train, generate_pokemon
generate_pokemon()
train(epochs=100000, batch_size=128, save_interval=500)
```
## example:
<p align="left">
  <img width="500" src="https://github.com/matan-chan/pokemon_DCGAN/blob/main/examples/example1.png?raw=true">
</p>

