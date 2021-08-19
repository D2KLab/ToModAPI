# Parameters

These are the parameters used for the experiment in the paper:

> Ismail Harrando, Pasquale Lisena and Raphaël Troncy. **Apples to Apples: A Systematic Evaluation of Topic Models**. In _Recent Advances in Natural Language Processing (RANLP)_, September 2021.

|       	|                   	| 20NG        	| AFP         	| Yahoo-Balanced| Yahoo-Unbalanced |
|-------	|-------------------	|-------------	|-------------	|-------------	|-------------	|
| **All**   | num topics        	| 20          	| 17          	| 26          	| 26          	|
| **LDA**  	| alpha             	| 0.1         	| 0.1         	| 10         	| 10         	|
|       	| beta              	| null        	| null        	| null        	| null        	|
|       	| random_seed       	| 5           	| 5          	| 5            	| 5           	|
|       	| iterations        	| 800         	| 1000         	| 1500        	| 1400         	|
|       	| optimize_interval 	| 10          	| 10           	| 10          	| 20          	|
|       	| topic_threshold   	| 0           	| 0            	| 0           	| 0           	|
| **LFTM**  | alpha             	| 1           	| 0.1          	| 1.0         	| 1.0       	|
|       	| beta              	| 1           	| 0.1          	| 0.1         	| 0.1         	|
|       	| lambda            	| 1           	| 1            	| 1           	| 1           	|
|       	| initer            	| 1000        	| 800          	| 500         	| 1000         	|
|       	| niter             	| 200         	| 200          	| 100         	| 100         	|
|       	| topn              	| 10          	| 10           	| 10          	| 10          	|
| **D2T**  	| batch_size        	| 6144        	| 6144         	| 6144        	| 3072        	|
|       	| n_epochs          	| 80          	| 15           	| 20          	| 40          	|
|       	| lr                	| 0.05        	| 0.04         	| 0.05        	| 0.1        	|
|       	| l1_doc            	| 0.000002    	| 0.000002     	| 0.000002    	| 0.000002    	|
|       	| l1_word           	| 0.000000015  	| 0.000000015 	| 0.000000015 	| 0.000000015 	|
|       	| word_dim          	| 0           	| 0           	| 0           	| 0           	|
| **GSDMM** | alpha             	| 0.1         	| 0.1         	| 1.0         	| 1.0         	|
|       	| beta              	| 0.1         	| 0.1         	| 1.0         	| 1.0         	|
|       	| n_iter            	| 15          	| 10          	| 10          	| 10          	|
| **CTM**   | bert_input_size       | 768        	| 768      	    | 768      	    | 768         	|
|       	| num_epochs            | 200         	| 100     	    | 25     	    | 25         	|
|       	| hidden_sizes          | (100,)       	| (100,)        | (200,)        | (200,)        |
|       	| batch_size           	| 200          	| 200          	| 200          	| 10          	|
|       	| dropout           	| 0.4          	| 0.4          	| 0.4          	| 10          	|
|       	| inference_type       	| 'combined' 	| 'combined'  	| 'combined'  	| 'combined'    |
| **PVTM**  | vector_size           | 50        	| 50            | 50            | 50         	|
|       	| hs                    | 0         	|  0      	    | 0      	    | 0.1         	|
|       	| dbow_words            | 1          	|  1        	|  1        	| 10          	|
|       	| dm                   	| 0          	|  0        	| 0         	| 10          	|
|       	| epochs       	        | 30 	        |  100        	| 100        	| 30          	|
|       	| window       	        | 20   	        |  20       	| 20       	    | 10          	|
|       	| seed       	        | 123 	        |  123       	| 123       	| 10          	|
|       	| min_count       	    | 5 	        |  5         	| 5         	| 10          	|
|       	| workers       	    | 5 	        |  5         	| 5         	| 10          	|
|       	| alpha       	        | 0.1    	    |  0.05         |  0.05         | 0.01          |
|       	| min_alpha       	    | 0.025 	    |  0.25        	| 0.25        	| 10          	|
|       	| random_state       	| 123 	        |  123         	| 123         	| 10          	|
|       	| covariance_type       | 'diag' 	    |  'diag'      	| 'diag'      	| 10          	|
| **HDP**   | max_chunks            | None        	| None          | None          | 0.1         	|
|       	| max_time              | None         	| None     	    | None     	    | 0.1         	|
|       	| chunksize             | 256         	| 256          	| 256          	| 10          	|
|       	| kappa                 | 1.0          	| 1.0          	| 1.0          	| 10          	|
|       	| tau       	        | 64.0 	        | 64.0        	| 64.0        	| 10          	|
|       	| K       	            | 15 	        | 15          	| 15          	| 10          	|
|       	| T       	            | 150 	        | 150          	| 150          	| 10          	|
|       	| alpha       	        | 1	            | 1         	| 1         	| 10          	|
|       	| gamma       	        | 1 	        | 1          	| 1          	| 10          	|
|       	| eta       	        | 0.1 	        | 0.1          	| 0.1          	| 0.1          	|
|       	| scale       	        | 2.0	        | 2.0          	| 2.0          	| 2.0          	|
|       	| var_converge       	| None 	        | None        	| None        	| 10          	|
|       	| random_state          | None 	        | None        	| None        	| 10          	|
| **LSI**   | use_tfidf             | False        	| True        	| True        	| True        	|
|       	| chunksize             | 20000         | 20000         | 20000         | 10          	|
|       	| decay                 | 1.0          	| 2.0          	| 2.0          	| 1.0          	|
|       	| distributed       	| False 	    | False      	| False      	| 10          	|
|       	| onepass       	    | True 	        | True       	| True       	| 10          	|
|       	| power_iters       	| 2 	        | 2         	| 2         	| 10          	|
|       	| extra_samples         | 100 	        | 100          	| 3000          | 3000        	|
| **NMF**   | passes                | 1        	    | 1        	    | 1        	    | 0.1         	|
|       	| kappa                 | 2.0         	| 2.0          	| 2.0          	| 10          	|
|       	| minimum_probability   | 0.001         | 0.01        	| 0.01        	| 10          	|
|       	| w_max_iter       	    | 200 	        | 200          	| 200          	| 10          	|
|       	| w_stop_condition      | 0.0001 	    | 0.0001      	| 0.0001      	| 10          	|
|       	| h_max_iter            | 50 	        | 50        	| 50        	| 10          	|
|       	| h_stop_condition      | 0.01 	        | 0.01       	| 0.01       	| 10          	|
|       	| eval_every            | 10 	        | 10          	| 10          	| 10          	|
|       	| normalize             | True 	        | True      	| True      	| 10          	|
|       	| random_state          | None 	        | None      	| None      	| 10          	|
