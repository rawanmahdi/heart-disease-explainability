# heart-disease-explainability #

## Project Scope ##
Over the past decade, ML has made clear it's profound use in a variety of fields in medicine, including mobile healthcare, that is apps that provide some kind of healthcare oriented service to it's user. 

Integrating predictive models into widley accessible platforms like health apps serves a high risk to public health and saftey if the model behaviour is not thorughouly analyzed and validated from both the technical and clinical standpoint. Although increasingly complex models - such as neural networks - provide greater preformance, their black box nature makes their decisions diffcult to interpret as opposed to simpler models such as decision trees. 

### SHAP - Shapley Additive Explainations ###
One of the most popular approaches for model explainability is SHAP, developed by Lundberg and Lee at the Microsoft research team. It's based on Loyde Shapley's coalitional game theory, where each player (in our context, feature) recieves a score reflecting thier proportion of contribution to the game (model output). The developed algorithm provides a local post hoc explanation. Each feature's shapley value can be obtained through the summation below:

![alt text](https://github.com/rawanmahdi/explainable-ai-heart/blob/main/img/shap-formula.png?raw=true)

Applying Kernel SHAP to my heart disease classifier, we can see some of the classifier's behaviour depicted through the trends in it's explanations. For starters, mean feature importances across 100 samples from the testing dataset are given in the bar graph below. 

![alt text](https://github.com/rawanmahdi/explainable-ai-heart/blob/main/img/shap-plots/indicators-plots/100-sample-equiv-bar.png?raw=true)

Age being the most important factor that influences an individual's risk of heart disease gives us the notion that the model is correctly interpreting the age feature. Another interesting graph we can obtain through the shap library depicts the effect of a feature's values. The first plot depicts a younger male,whose relativley healthy. 

![alt text](https://github.com/rawanmahdi/explainable-ai-heart/blob/main/img/shap-plots/indicators-plots/200-sample-similarity-younger-healthy-male.png?raw=true)
The model predicted that this individual has a 10% risk, attributing his younger age and strong overall health to lowering his risk, and his sex to raising it. We can compare this to an older female who also happens to smoke:
![alt text](https://github.com/rawanmahdi/explainable-ai-heart/blob/main/img/shap-plots/indicators-plots/200-sample-similarity-older-smoking-female.png?raw=true)
This individual was predicted to have a much higher risk of 61%, and according to the shapley values, her age and the fact that she smokes, and has diabetes increased her risk the most out of all other factors, whereas her sex decreased it. 

Although these results are exciting, in that they seemingly clinically validate the behaviour of our model, a major drawback exists in this specific implementation of shapley game theory for explainability. Kernel SHAP makes the major assumption that our features are independant, and in the process of forming feature coalitions, it passes in potentially unrealistic feature value combinations, making it unreliable in the context of clinical validation, and even model debugging.

### SHAPR - SHAP for dependant features ###
In an attempt to get more reliable results, I found an implementation of shapley values that don't make the assumption of independance between features. I preformed the same explanations as above using shapr, developed by Aas et al. Below is the corresponding mean absolute feature importance. 

![alt text](https://github.com/rawanmahdi/explainable-ai-heart/blob/main/img/shapr-plots/100-samples-bar.png?raw=true)
### Counterfactual Explanations ###

## Privacy Preservation aproach ##
An important consideration when deploying health-based models is how we can maintain high preformance while perserving the patient's privacy. Adopting federated learning and on device training/explanations can help address security and privacy concerns. 
![alt text](https://github.com/rawanmahdi/explainable-ai-heart/blob/main/img/serving-vs-local.png?raw=true)

## Dataset Links ##
**14 attributes of heart disease**: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset 

**Personal Indicators of Heart Disease**: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease 

## Resources and Tutorials ##
### **Explainable AI** ###
**IBM's AIX360 Explainability Methods + Use Cases**
![alt text](https://github.com/rawanmahdi/explainable-ai-heart/blob/main/img/methods-choice.gif?raw=true)
http://aix360.mybluemix.net/resources#guidance 

**DeepFindr's Tutorials**

https://www.youtube.com/playlist?list=PLV8yxwGOxvvovp-j6ztxhF3QcKXT6vORU
### **Deep Learning Model** ###
**Classify structured data using Keras preprocessing layers**
 https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers

**Classification on imbalanced data**
 https://www.tensorflow.org/tutorials/structured_data/imbalanced_data

**Undersampling imbalanced dataset using imblearn**
https://www.kaggle.com/code/residentmario/undersampling-and-oversampling-imbalanced-data 

 **Heart Disease Diagnostic Model**
 https://towardsdatascience.com/heart-disease-prediction-in-tensorflow-2-tensorflow-for-hackers-part-ii-378eef0400ee 
