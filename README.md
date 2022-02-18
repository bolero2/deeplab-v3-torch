# deeplab-v3-torch
Easy-to-use VGGNet


## Preparation
1) Prepare your **Dataset**.
2) Modify ``setting.yaml`` file. (Check out the `# write here` comments section.)
3) Run ``python main.py``.


## Core functions
1) ``model.fit(x, y, validation_data, epochs=30, batch_size=4)``   
    : Training function is reconstructed like **tensorflow-keras** style.
  
2) ``model.evaluate(model, dataloader, valid_iter=1, batch_size=1, criterion=None)``

3) ``model.predict(test_images, use_cpu=False)``
