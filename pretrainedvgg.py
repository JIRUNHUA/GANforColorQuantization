def discriminator(input):
	def test_load():
	    data_path = './/vgg16_pretrain//vgg16.npy'
	    data_dict = np.load(data_path, encoding='latin1').item()
	    keys = sorted(data_dict.keys())
	    for key in keys:
	        weights = data_dict[key][0]
	        biases = data_dict[key][1]
	 
	def conv(layer_name,input,out_channels,is_pretrained=True):
		'''
		Args:
			layer_name:	e.h. conv1, pool
			input:	input tensor,[batch, height, wights, channels]
			out_channels:	number of output channel
			is_pretrained:	 if load pretrained parameters, freeze all conv layers. 
			Depending on different situations, you can just set part of conv layers to be freezed.
	        the parameters of freezed layers will not change when training.
		return:	4-D tensor
		'''
		in_channels=input.get_shape()[-1]
		with tf.variable_scope(layer_name):
			weights=tf.get_variable(name="weights",trainable=is_pretrained,shape=[3,3,in_channels,out_channels],initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
			biases=tf.get_variable(name="biases",trainable=is_pretrained,shape=[out_channels],initializer=tf.constant_initializer(0.1))
		    net=tf.nn.conv2d(input,weights,[1,1,1,1], padding="SAME",name="conv")
		    net=tf.nn.bias_add(input,biases,name="bias_add")	   
		    net=tf.nn.relu(input,name="relu")   
		    return net  

	def pool(layer_name,input):
		with variable_scope(layer_name):	
			return tf.nn.mainput_pool(net,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME"，name=layer_name)

	def batch_norm(input):
		'''
		Batch Normalization 
		'''
		'''
			channels = input.get_shape()[3]
	        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
	        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
	        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
		'''
	    epsilon = 1e-3
	    batch_mean,batch_var=tf.nn.moment(input,[0])
	    normalized=tf.nn.batch_normalization(input,mean=batch_mean,cariance=batch_var,offset=None,scale=None,variance_epsilon=epsilon)
	    return normalized

	with variable_scope("VGG16_conv"):
		# block1 : [batch, 256, 256, 3] => [batch, 128, 128, 64]  
		conv1_1_relu=conv('conv1_1',input,64)
		conv1_2_relu=conv('conv1_2',conv1_1_relu,64)
		pool1=pool('pool1',conv1_2_relu)		
		# block2 : [batch, 128, 128, 64] => [batch, 64, 64, 128]  
		conv2_1_relu=conv('conv2_1',pool1,128)
		conv2_2_relu=conv('conv2_2',conv2_1_relu,128)
		pool2=pool('pool2',conv2_2_relu)
		# block3 : [batch, 64, 64, 128] => [batch, 32, 32, 256]  
		conv3_1_relu=conv('conv3_1',pool2,256)
		conv3_2_relu=conv('conv3_2',conv3_1_relu,256)
		conv3_3_relu=conv('conv3_3',conv3_2_relu,256)
		pool3=pool('pool3',conv3_3_relu)
		# block4 : [batch, 32, 32, 256] => [batch, 16, 16, 512]    
		conv4_1_relu=conv('conv4_1',pool3,512)
		conv4_2_relu=conv('conv4_2',conv4_1_relu,512)
		conv4_3_relu=conv('conv4_3',conv4_2_relu,512)
		pool4=pool('pool4',conv4_3_relu)
		#block5 : [batch, 16, 16, 512] => [batch, 8, 8, 512]   
		conv5_1_relu=conv('conv5_1',pool4,512)
		conv5_2_relu=conv('conv5_2',conv5_1_relu,512)
		conv5_3_relu=conv('conv5_3',conv5_2_relu,512)
		pool5=pool('pool5',conv5_3_relu)
		
	with tf.variable_scope('FC_Softmax'):	
		#block6 : [batch, 8, 8, 512] => [batch, 4096]
		with tf.variable_scope('FC6') as scope:
			shape=int(np.prod(pool5.get_shape()[1:]))
		    weight=tf.Variable(tf.truncated_normal([shape,4096],dtype=tf.float32,stddev=1e-1))
		    bias=tf.Variable(tf.constant(1.0,shape=[4096],dtype=tf.float32))
		    flat=tf.reshape(pool5,[-1,shape])
		    fc6=tf.nn.bias_add(tf.matmul(flat,weight),bias)
		    fc6=tf.nn.relu(fc6) 
		    fc6=batch_norm(fc6)
		#block7 : [batch, 4096] => [batch, 1024]
		with tf.variable_scope('FC7') as scope:	
		    weight=tf.Variable(tf.truncated_normal([4096,1024],dtype=tf.float32,stddev=1e-1))
		    bias=tf.Variable(tf.constant(1.0,shape=[1024],dtype=tf.float32))	   
		    fc7=tf.nn.bias_add(tf.matmul(fc6,weight),bias)
		    fc7=tf.nn.relu(fc7) 
		    fc7=batch_norm(fc7) 
		#block8 : [batch, 1024] => [batch, 256]
		with tf.variable_scope('FC8') as scope:	
		    weight=tf.Variable(tf.truncated_normal([1024,256],dtype=tf.float32,stddev=1e-1))
		    bias=tf.Variable(tf.constant(1.0,shape=[256],dtype=tf.float32))	   
		    fc8=tf.nn.bias_add(tf.matmul(fc7,weight),bias)
		    fc8=tf.nn.relu(fc8) 
		    fc8=batch_norm(fc8) 
		#block8 : [batch, 256] => [batch, 2]
		with tf.variable_scope('Softmax9') as scope:	
		    weight=tf.Variable(tf.truncated_normal([256,2],dtype=tf.float32,stddev=1e-1))
		    bias=tf.Variable(tf.constant(1.0,shape=[2],dtype=tf.float32))	
		    softmax = tf.add(tf.matmul(fc8,weights),biases)   

	return softmax






