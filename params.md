# Parameters

These are the parameters used for the experiment in the paper.

|       	|                   	| 20NG        	| AFP         	| TED         	|
|-------	|-------------------	|-------------	|-------------	|-------------	|
| **All**   | num topics        	| 20          	| 17          	| 25          	|
| **LDA**  	| alpha             	| 0.1         	| 0.1         	| 0.1         	|
|       	| beta              	| null        	| null        	| null        	|
|       	| random_seed       	| 5           	| 5           	| 5           	|
|       	| iterations        	| 800         	| 1000        	| 800         	|
|       	| optimize_interval 	| 10          	| 10          	| 10          	|
|       	| topic_threshold   	| 0           	| 0           	| 0           	|
| **LFTM**  | alpha             	| 1           	| 0.1         	| 0.1         	|
|       	| beta              	| 1           	| 0.1         	| 0.1         	|
|       	| lambda            	| 1           	| 1           	| 1           	|
|       	| initer            	| 1000        	| 800         	| 700         	|
|       	| niter             	| 200         	| 200         	| 100         	|
|       	| topn              	| 10          	| 10          	| 10          	|
| **D2T**  	| batch_size        	| 6144        	| 6144        	| 6144        	|
|       	| n_epochs          	| 80          	| 15          	| 50          	|
|       	| lr                	| 0,05        	| 0,04        	| 0,05        	|
|       	| l1_doc            	| 0,000002    	| 0,000002    	| 0,000002    	|
|       	| l1_word           	| 0,000000015 	| 0,000000015 	| 0,000000015 	|
|       	| word_dim          	| 0           	| 0           	| 0           	|
| **GSDMM** | alpha             	| 0.1         	| 0.1         	| 0.1         	|
|       	| beta              	| 0.1         	| 0.1         	| 0.1         	|
|       	| n_iter            	| 15          	| 10          	| 10          	|
| **CTM**   | bert_input_size       | 512        	| 512      	    | 0.1         	|
|       	| num_epochs            | 200         	| 100     	    | 0.1         	|
|       	| hidden_sizes          | (100,)       	| (100,)         	| 10          	|
|       	| batch_size           	| 200          	| 200          	| 10          	|
|       	| dropout           	| 0.4          	| 0.4          	| 10          	|
|       	| inference_type       	| 'combined' 	| 'combined'  	| 10          	|
| **PVTM**  | vector_size           | 50        	| 50            | 0.1         	|
|       	| hs                    | 0         	|  0      	    | 0.1         	|
|       	| dbow_words            | 1          	|  1        	| 10          	|
|       	| dm                   	| 0          	|  0        	| 10          	|
|       	| epochs       	        | 30 	        |  100        	| 10          	|
|       	| window       	        | 20   	        |  20       	| 10          	|
|       	| seed       	        | 123 	        |  123       	| 10          	|
|       	| min_count       	    | 5 	        |  5         	| 10          	|
|       	| workers       	    | 5 	        |  5         	| 10          	|
|       	| alpha       	        | 0.1    	    |  0.05         | 10          	|
|       	| min_alpha       	    | 0.025 	    |  0.25        	| 10          	|
|       	| random_state       	| 123 	        |  123         	| 10          	|
|       	| covariance_type       | 'diag' 	    |  'diag'      	| 10          	|
| **HDP**   | max_chunks            | None        	| None          | 0.1         	|
|       	| max_time              | None         	| None     	    | 0.1         	|
|       	| chunksize             | 256         	| 256          	| 10          	|
|       	| kappa                 | 1.0          	| 1.0          	| 10          	|
|       	| tau       	        | 64.0 	        | 64.0        	| 10          	|
|       	| K       	            | 15 	        | 15          	| 10          	|
|       	| T       	            | 150 	        | 150          	| 10          	|
|       	| alpha       	        | 1	            | 1         	| 10          	|
|       	| gamma       	        | 1 	        | 1          	| 10          	|
|       	| eta       	        | 0.1 	        | 0.1          	| 10          	|
|       	| scale       	        | 2.0	        | 2.0          	| 10          	|
|       	| var_converge       	| None 	        | None        	| 10          	|
|       	| random_state          | None 	        | None        	| 10          	|
| **LSI**   | use_tfidf             | False        	| True        	| 0.1         	|
|       	| chunksize             | 20000         | 20000         | 10          	|
|       	| decay                 | 1.0          	| 2.0          	| 10          	|
|       	| distributed       	| False 	    | False      	| 10          	|
|       	| onepass       	    | True 	        | True       	| 10          	|
|       	| power_iters       	| 2 	        | 2         	| 10          	|
|       	| extra_samples         | 100 	        | 100          	| 10          	|
| **NMF**   | passes                | 1        	    | 1        	    | 0.1         	|
|       	| kappa                 | 2.0         	| 2.0          	| 10          	|
|       	| minimum_probability   | 0.001         | 0.01        	| 10          	|
|       	| w_max_iter       	    | 200 	        | 200          	| 10          	|
|       	| w_stop_condition      | 0.0001 	    | 0.0001      	| 10          	|
|       	| h_max_iter            | 50 	        | 50        	| 10          	|
|       	| h_stop_condition      | 0.01 	        | 0.01       	| 10          	|
|       	| eval_every            | 10 	        | 10          	| 10          	|
|       	| normalize             | True 	        | True      	| 10          	|
|       	| random_state          | None 	        | None      	| 10          	|
