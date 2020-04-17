# Parameters

These are the parameters used for the experiment in the paper.

The models in `models` folder have been computed using the params in the TED column, over the [TED Talks dataset]('../data/data.txt').

Are you using Postman? See the requests in this [collection](https://documenter.getpostman.com/view/1103941/Szf53pCA?version=latest).

|       	|                   	| 20NG        	| AFP         	| TED         	|
|-------	|-------------------	|-------------	|-------------	|-------------	|
| **All**   | num topics        	| 20          	| 17          	| 25          	|
| **LDA**  	| alpha             	| 0,1         	| 0,1         	| 0,1         	|
|       	| beta              	| null        	| null        	| null        	|
|       	| random_seed       	| 5           	| 5           	| 5           	|
|       	| iterations        	| 800         	| 1000        	| 800         	|
|       	| optimize_interval 	| 10          	| 10          	| 10          	|
|       	| topic_threshold   	| 0           	| 0           	| 0           	|
| **LFTM**  | alpha             	| 1           	| 0,1         	| 0,1         	|
|       	| beta              	| 1           	| 0,1         	| 0,1         	|
|       	| lambda            	| 1           	| 1           	| 1           	|
|       	| initer            	| 1000        	| 800         	| 700         	|
|       	| niter             	| 200         	| 200         	| 100         	|
|       	| topn              	| 10          	| 10          	| 10          	|
| **NTM**  	| batch_size        	| 6144        	| 6144        	| 6144        	|
|       	| n_epochs          	| 80          	| 15          	| 50          	|
|       	| lr                	| 0,05        	| 0,04        	| 0,05        	|
|       	| l1_doc            	| 0,000002    	| 0,000002    	| 0,000002    	|
|       	| l1_word           	| 0,000000015 	| 0,000000015 	| 0,000000015 	|
|       	| word_dim          	| 0           	| 0           	| 0           	|
| **GSDMM** | alpha             	| 0,1         	| 0,1         	| 0,1         	|
|       	| beta              	| 0,1         	| 0,1         	| 0,1         	|
|       	| n_iter            	| 10          	| 10          	| 10          	|
