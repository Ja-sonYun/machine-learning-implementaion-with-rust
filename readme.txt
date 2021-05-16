
,-------------------------------------------------------------------------
| 1  let input = vec![0.1, 0.2, 0.5, 0.8];
| 2  let output = vec![0.56, 0.5, 0.93, 0.6, 0.23, 0.654, 0.2, 0.1];
| 3  let mut new_model = Model::new("test", &MSE, &he_initializer);
| 4  new_model.add_layer(&SGD, 4, 4, IN(input), &Sigmoid, Some("input"));
| 5  new_model.add_layer(&SGD, 4, 4, HIDDEN, &Sigmoid, Some("input"));
| 6  new_model.add_layer(&SGD, 4, 8, HIDDEN, &Sigmoid, Some("input"));
| 7  new_model.add_layer(&SGD, 8, 0, OUT(output), &Sigmoid, Some("output"));
| 8  new_model.train(100, 0.2, true, 20);
|------------OUTPUT----------------------------------------------------
|  0 step, loss: 0.8380310284839548
| 20 step, loss: 0.573779936458767
| 40 step, loss: 0.27056526242049006
| 60 step, loss: 0.12288962559780128
| 80 step, loss: 0.06884584321349119
`---------------------------------------------------------------------



TODO : 
 - Matrix
   |- Transpose
   |- Concatenate and other calculations
   `- Non element wise multiply ?

 - Layer
   |- SGD
   `- Use matrix
      `- weight initializers
  
 - PNG decoder
   `- IDAT decode algorithm
   
   
DONE :
 - Matrix
   |- Use single vector for N dimensions
   |------------------------------------------------------------
   |    EXAMPLES                 `- Matrix { _ndarray: Vec<T>, ...}
   | dimension => [ 3 ]           set/get => [ 2 ]
   |              x-'                          '-x
   | dimension => [ 3, 4 ]        set/get => [ 2, 2 ]
   |              y-'  |                     y-'   `-x
   |                   `-x
   | dimension => [ 3, 4, 5 ]     set/get => [ 2, 2, 2 ]
   |              y-'  |  '-z                y-'  |  '-z
   |                   `-x                      x-'
   | dimension => [ 3, 4, 5, 6 ]  set/get => [ 2, 2, 2, 2 ]
   |              y-'  |  '-z `-k            y-'  |  |   `-k
   |                   `-x                      x-'  `-z
   |------------------------------------------------------------
   | 1  let A = Matrix::<f64>::ones(vec![1, 5, 4, 5, 5, 5]);
   | 2  A.set(vec![0, 3, 2, 2, 2, 2], 3.);
   | 3  assert_eq!(A.get(vec![0, 3, 2, 2, 2, 2]), 3.);
   |------------------------------------------------------------
   |- Element wise calculation
   |- Create Matrix from png Image -> vec![height, width, RGB]
   |   `- currently using png crate to decode png,  TODO: IDAT algorithm
   `- Create, Calculation with scalar, get, set, etc
 
 - Model
   |- Designable Model Struct 
   |             `- { name, LayerObj, layer_actions<layer, activation>,
   |                  layer_size, lr, cost, cost_function, w_initializer }
   |- create model with
   |-------------------------------------------------------------------------
   | 1 `-  let mut new_model = Model::new("test", &MSE, &he_initializer);
   |-------------------------------------------------------------------------
   |                                 ,-Input dimension(4 neurons wil generated)
   |- add input layer with           |  ,-Output dimension
   |-------------------------------------------------------------------------
   | 2 `-  new_model.add_layer(&SGD, 4, 4, IN(input), &Sigmoid, Some("input"));
   |-------------------------------------------------------------------------
   |- add hidden layer with          ,--'
   |-------------------------------------------------------------------------
   | 3 `-  new_model.add_layer(&SGD, 4, 8, HIDDEN, &Sigmoid, Some("input"));
   |-------------------------------------------------------------------------
   |- add output layer with          ,--'
   |-------------------------------------------------------------------------
   | 4 `-  new_model.add_layer(&SGD, 8, 0, OUT(output), &Sigmoid, Some("output"));
   |-------------------------------------------------------------------------
   |- start train with
   | 5 `-  new_model.train(500, 0.2, true, 10); // where 500 is steps, 0.2 is learning rate, log, and log interval.
   `-------------------------------------------------------------------------

   

 - Neurons
   `- each neurons has theirs initializer, local loss, bias, output, weights, and input
   
 - Layer
   |- back propagaion, feed forward traits that executed on model object
   |- traits are stored at LayerObj
   |                       `- { name, neurons, fan_in, fan_out, _type }  
   |- not finished..?
   
 - Activation, Cost, Initializer
   |- back propagaion, feed forward traits
   |-------------------------------------------------------------------------
   | 1  impl<'lng> Activation<'lng> for Leaky_ReLU {
   | 2      fn feed_forward(&self, x:f64) -> f64 { }
   | 3      fn back_propagation(&self, x:f64) -> f64 { }
   | 4  }
   `-------------------------------------------------------------------------


  
