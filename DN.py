'''
VGG16 :
> conv1_1 > relu1_1 > conv1_2 > relu1_2 > pool1
===
> conv2_1 > relu2_1 > conv2_2 > relu2_2 > pool2
===
> conv3_1 > relu3_1 > conv3_2 > relu3_2 > conv3_3 > relu3_3 > pool3
===
> conv4_1 > relu4_1 > conv4_2 > relu4_2 > conv4_3 > relu4_3 > pool4
===
> conv5_1 > relu5_1 > conv5_2 > relu5_2 > conv5_3 > relu5_3 > pool5
===
> FC-4096 > FC-4096 > FC-100 > soft-max

'''


def discriminator(input1, input2):
	# 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
	input=tf.concat([input1,input2],axis=3)

    # block1 : [batch, 256, 256, 6] => [batch, 128, 128, 64]  
    conv1_1_relu=conv(input,64)
    conv1_2_relu=conv(conv1_1_relu,64)
    pool1=pool(conv1_2_relu)
    	
    # block2 : [batch, 128, 128, 64] => [batch, 64, 64, 128]  
    conv2_1_relu=conv(pool1,128)
    conv2_2_relu=conv(conv2_1_relu,128)
	pool2=pool(conv2_2_relu)

    # block3 : [batch, 64, 64, 128] => [batch, 32, 32, 256]  
    conv3_1_relu=conv(pool2,256)
    conv3_2_relu=conv(conv3_1_relu,256)
    conv3_3_relu=conv(conv3_2_relu,256)
    pool3=pool(conv3_3_relu)

    # block4 : [batch, 32, 32, 256] => [batch, 16, 16, 512]    
    conv4_1_relu=conv(pool3,512)
    conv4_2_relu=conv(conv4_1_relu,512)
    conv4_3_relu=conv(conv4_2_relu,512)
    pool4=pool(conv4_3_relu)

    #block5 : [batch, 16, 16, 512] => [batch, 8, 8, 512]   
    conv5_1_relu=conv(pool4,512)
    conv5_2_relu=conv(conv5_1_relu,512)
    conv5_3_relu=conv(conv5_2_relu,512)
    pool5=pool(conv5_3_relu)

 #    #block6 :FC layers
	# fc1=FC_layer(pool5,4096)
	# fc2=FC_layer(fc1,4096)
	# fc3=FC_layer(fc2,1000)
        
	# def FC_layer(input,out_channels):
	# 	if len(input.get_shape)==4:			
	#     	shape=int(np.prod(input.get_shape()[1:]))
	#     else:
	#     	shape=int(np.prod(input.get_shape()[-1]))
	#     weight=tf.Variable(tf.truncated_normal([shape,out_channels],dtype=tf.float32,stddev=1e-1))
	#     bias=tf.Variable(tf.constant(1.0,shape=[out_channels],dtype=tf.float32))
	#     flat=tf.reshape(input,[-1,shape])
	#     fc=tf.nn.bias_add(tf.matmul(flat,weight),bias)
	#     fc=tf.nn.relu(fc)   
	#     return fc

	result= tf.sigmoid(pool5)
	return result

	def kernel_init(in_channels,out_channels):
		weights=tf.Variable(tf.truncated_normal([3,3,in_channels,out_channels],dtype=tf.float32,stddev=1e-1))
		biases=tf.Variable(tf.constant(0.0,shape=[out_channels],dtype=tf.float32))
		return weights,biases

	def conv(net,out_channels,relu=True):
		in_channels=net.get_shape()[3]
		weights,biases=kernel_init(in_channels,out_channels)
	    conv=tf.nn.conv2d(net,weights,[1,1,1,1], padding="SAME")
	    conv=tf.nn.bias_add(conv,biases)	   
	    if relu:
	        conv=tf.nn.relu(namedtuple)   
	    return conv   

	def pool(net):
		return tf.nn.max_pool(net,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")



