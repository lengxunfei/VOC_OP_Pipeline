#  **Machine Learning-Based Prediction Model for Insect Olfactory Protein-Volatile Organic Ligand Binding**

The VOC-OP binding activity classification system is a computational pipeline based on machine learning, used to predict the binding activity between volatile organic compounds (VOCs) and odor proteins (OPs). This system integrates multiple machine learning algorithms and advanced chemical informatics techniques, providing researchers with accurate classification predictions of binding activities. Integrated explainable AI (XAI) and t-SNE visualization components to reveal molecular interaction mechanisms and identify key molecular descriptors governing insect olfactory recognition.

<img width="1263" height="1052" alt="Graphical Abstract" src="https://github.com/user-attachments/assets/8aee991a-675f-4361-94b5-f9626c6ec963" />



## **Requirements**  

python == 3.9  
RDKit == 2023.3.3  
scikit-learn == 1.6.1  
Pandas == 2.2.3  
NumPy == 1.26.4  
PubChemPy == 1.0.4  
RDKit == 2023.3.3  
Matplotlib == 3.9.1  
Seaborn  == 0.13.2  
SHAP == 0.45.1
LIME == 0.2.0.1  
MySQL-connector-Python == 8.3.0

## **Run Pipeline**  

The system offers a one-click operation mode. Simply execute the following command to start the complete data processing, model training and result analysis process: "python VOC_OP_Pipeline.py". If using an IDE or code editor, you can directly open and run the script.

## **The dataset files**  

**Users can view the relevant content of the database on which the pipeline operates at any time.**

•	All_Compound_Descriptor.xlsx: All descriptor data of all compounds. 
•	Compound_info_1.xlsx: Compound Information File 1.  
•	Compound_info_2.xlsx: Compound Information File 2.  
•	Compound_OP_binding.xlsx: Data on the combination of the compound with OP.  
•	Descriptor_vif_results.xlsx: Analysis results of descriptor VIF.  
•	OP_info.xlsx: Odorant protein information.  

## **Output files**  

**After the training is completed, corresponding folders will be generated under the "system_result" directory based on the research OP taxon and training parameters, such as "Lepidoptera_10.0_SVM_5".**

### 1.  ../results/ （Results files）

•	Confusion_Matrix.png:   
•	PR_curve.png:   
•	ROC_curve.png:   
•	results.json: Outcome indicators.
•	scaler.pkl: Data Normalizer.  
•	selected_features.xlsx: List of selected features.  
•	SHAP_Heatmap.png:   
•	SHAP_Summary_Bar_Plot.png: 
•	SHAP_Summary_Violin_Plot.png: S
•	tsne_plot_all.png: Complete data t-SNE visualization.  
•	tsne_plot_selected.png: Visualization using t-SNE after feature selection. 

### 2.  ../train_model/ （Saved model）

•	model.pkl: The trained machine learning model, used for prediction.

### 3.  ../  (Intermediate processing file)

•	All_Compound_Descriptor_original.xlsx: Original descriptor calculation result.
•	All_Compound_Descriptor_remove0.xlsx: Remove the calculation results of descriptors with a variance of 0.
•	All_Compound_Descriptor_remove0_removeVif.xlsx: Further eliminate the calculation results of descriptors with VIF larger than 5.	  
•	All_Compound_Descriptor_remove0_removeVif_Standard.xlsx: The calculated results of the descriptors after further standardization.
•	prefix.xlsx: The VOC that effectively combines with the target OP taxon and its binding value.
•	prefix_label.xlsx: In the target OP taxon, the active/inactive marking results of VOC.
•	vif_results.xlsx: The calculation result of the descriptor's vif.
