import json
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.metrics import roc_auc_score, average_precision_score
import tkinter as tk
from tkinter import  ttk
import  random
import lime
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer
import shap
from scipy.stats import loguniform, uniform
from scipy.stats import randint
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, VarianceThreshold
from sklearn.linear_model import SGDClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle
from PIL import Image, ImageTk
import logging
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, matthews_corrcoef, recall_score, f1_score, precision_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pubchempy as pcp
import requests
from rdkit import Chem
import time

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='project_log.log',
                    filemode='w')

logger = logging.getLogger(__name__)

def enhanced_smiles_retrieval(identifier):


    if is_valid_smiles(identifier):
        return identifier

    smiles = None

    # 1. PubChem
    smiles = try_pubchem(identifier)
    if smiles:
        print("The SMILES format of the compound is ", smiles)
        return smiles

    # 2. CIR
    smiles = try_cir(identifier)
    if smiles:
        print("The SMILES format of the compound is ", smiles)
        return smiles

    # 3. ChemSpider
    smiles = try_chemspider(identifier)
    if smiles:
        print("The SMILES format of the compound is： ", smiles)
        return smiles

    # 4. OPSIN
    smiles = try_opsin(identifier)
    if smiles:
        print("The SMILES format of the compound is： ",smiles)
        print("-----------------------------------")
        return smiles


    print(f"Unable to find the compound '{identifier}' smiles")
    return None


def is_valid_smiles(smiles):

    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def try_pubchem(identifier):

    try:
        print("Try to obtain the channel： pubchem")
        compounds = pcp.get_compounds(identifier, 'name')
        time.sleep(0.2)

        if not compounds:
            compounds = pcp.get_compounds(identifier, 'cas')
            time.sleep(0.2)


        if compounds:

            return compounds[0].canonical_smiles
    except Exception as e:
        print(f"PubChem failed: {e}")
    return None


def try_cir(identifier):

    try:
        print("Try to obtain the channel： cir")
        url = f"https://cactus.nci.nih.gov/chemical/structure/{identifier}/smiles"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.text.strip()
    except Exception as e:
        print(f"CIR failed: {e}")
    return None


def try_chemspider(identifier):

    print("Try to obtain the channel： chemspider")

    try:

        search_url = f"http://www.chemspider.com/Search.asmx/SimpleSearch?query={identifier}"
        response = requests.get(search_url)

        if response.status_code == 200:

            pass
    except Exception as e:
        print(f"ChemSpider failed: {e}")
    return None


def try_opsin(identifier):

    print("Try to obtain the channel： OPSIN")
    try:
        # OPSIN REST API
        url = f"https://opsin.ch.cam.ac.uk/opsin/{identifier}.json"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            return data.get("smiles")
    except Exception as e:
        print(f"OPSIN failed: {e}")
    return None

def name_or_cas_to_smiles(identifier):


    compound = pcp.get_compounds(identifier, 'name')
    time.sleep(0.2)

    if not compound:

        compound = pcp.get_compounds(identifier, 'cas')
        time.sleep(0.2)
    if compound:
        print(f"SMILES for '{identifier}': {compound[0].canonical_smiles}")
        return compound[0].canonical_smiles
    else:
        print(f"No compound found for '{identifier}'")
        return None


def compute_descriptors(smiles, feature_names):

    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(feature_names)

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES string")

    values = calculator.CalcDescriptors(mol)

    descriptors = {name: val for name, val in zip(feature_names, values) if not np.isnan(val)}


    return descriptors


class WelcomeApp:
    def __init__(self, master):
        self.master = master
        master.title("Welcome")
        master.geometry("800x600")
        master.configure(bg="#4CAF50")

        # Load the background image
        self.bg_image = Image.open("welcome_background.jpeg")
        self.bg_image = self.bg_image.resize((800, 600), Image.Resampling.LANCZOS)
        self.bg_image = ImageTk.PhotoImage(self.bg_image)

        # Create a label to hold the background image
        self.bg_label = tk.Label(master, image=self.bg_image)
        self.bg_label.place(relwidth=1, relheight=1)

        self.welcome_label = tk.Label(master, text="Welcome to use the VOC-OP \n binding activity classification system", font=("Helvetica", 28, "bold"), bg="#4CAF50", fg="white")
        self.welcome_label.place(relx=0.5, rely=0.5, anchor='center')

        master.after(3000, self.open_main_app)

    def open_main_app(self):
        self.master.destroy()
        root = tk.Tk()
        app = MainApp(root)
        root.mainloop()

class MainApp:
    def __init__(self, master):
        self.master = master
        master.title("VOC-OP combined with active classification system")
        master.geometry("800x600")
        master.configure(bg="#f0f0f0")

        self.center_frame = tk.Frame(master, bg="#f0f0f0")
        self.center_frame.place(relx=0.5, rely=0.5, anchor='center')

        self.welcome_label = tk.Label(self.center_frame, text="Welcome to the system!", font=("Helvetica", 24, "bold"), bg="#f0f0f0")
        self.welcome_label.grid(row=0, column=0, columnspan=2, pady=20)

        self.label1 = tk.Label(self.center_frame, text="Enter the order to which the olfactory protein belongs : ", font=('Helvetica', 12), bg="#f0f0f0")
        self.label1.grid(row=1, column=0, padx=10, pady=10, sticky='e')
        self.algo_var1 = tk.StringVar(master)
        self.algo_var1.set("Lepidoptera")
        self.algo_menu1 = ttk.Combobox(self.center_frame, textvariable=self.algo_var1, font=('Helvetica', 12), width=28)
        self.algo_menu1['values'] = (
            "Diptera", "Hymenoptera", "Hemiptera", "Lepidoptera", "Coleoptera",
            "Neuroptera","Thysanoptera", "Orthoptera", "Blattodea"
            )
        self.algo_menu1.grid(row=1, column=1, padx=10, pady=10, sticky='w')

        self.label2 = tk.Label(self.center_frame, text="Select the classification threshold for binding activity (0 - 100) : ", font=('Helvetica', 12), bg="#f0f0f0")
        self.label2.grid(row=2, column=0, padx=10, pady=10, sticky='e')
        self.algo_var2 = tk.StringVar(master)
        self.algo_var2.set("10")
        self.algo_menu2 = ttk.Combobox(self.center_frame, textvariable=self.algo_var2, font=('Helvetica', 12), width=28)
        self.algo_menu2['values'] = ("10", "20", "30",  "40", "50", "60","70","80","90")
        self.algo_menu2.grid(row=2, column=1, padx=10, pady=10, sticky='w')

        self.label3 = tk.Label(self.center_frame, text="Select the algorithm for model training : ", font=('Helvetica', 12), bg="#f0f0f0")
        self.label3.grid(row=3, column=0, padx=10, pady=10, sticky='e')
        self.algo_var3 = tk.StringVar(master)
        self.algo_var3.set("SVM")
        self.algo_menu3 = ttk.Combobox(self.center_frame, textvariable=self.algo_var3, font=('Helvetica', 12), width=28)
        self.algo_menu3['values'] = ("SVM", "KNN", "Random Forest",  "GBDT", "SGD", "GNB")
        self.algo_menu3.grid(row=3, column=1, padx=10, pady=10, sticky='w')

        self.label4 = tk.Label(self.center_frame, text="Enter the number of cross-validation folds (k > 0) ：", font=('Helvetica', 12), bg="#f0f0f0")
        self.label4.grid(row=4, column=0, padx=10, pady=10, sticky='e')
        self.algo_var4 = tk.StringVar(master)
        self.algo_var4.set("5")
        self.algo_menu4 = ttk.Combobox(self.center_frame, textvariable=self.algo_var4, font=('Helvetica', 12), width=28)
        self.algo_menu4['values'] = ("1", "2", "5", "10")
        self.algo_menu4.grid(row=4, column=1, padx=10, pady=10, sticky='w')

        main_button_width = 20

        self.train_button = tk.Button(self.center_frame, text="Start training", font=('Helvetica', 14, 'bold'), bg="#4CAF50",
                                      fg="white", command=self.start_training, width=main_button_width)
        self.train_button.grid(row=5, column=0, columnspan=2, pady=20)

        self.predict_button = tk.Button(self.center_frame, text="Predict function", font=('Helvetica', 14, 'bold'), bg="#4CAF50",
                                        fg="white", command=self.open_predict_window, width=main_button_width)
        self.predict_button.grid(row=6, column=0, columnspan=2, pady=20)

        self.open_button = tk.Button(self.center_frame, text="View the system database", command=self.open_new_window,
                                     font=('Helvetica', 14, 'bold'), bg="#4CAF50", fg="white", width=main_button_width)
        self.open_button.grid(row=7, column=0, columnspan=2, pady=20)

    def open_new_window(self):
        new_window = tk.Toplevel(self.master)
        new_app = DatabaseViewer(new_window)

    def open_predict_window(self):
        predict_window = tk.Toplevel(self.master)
        new_predict = PredictWindow(predict_window)


    def start_training(self):
        op_group = self.algo_var1.get()
        threshold = self.algo_menu2.get()
        algorithm = self.algo_var3.get()
        k_folds = self.algo_menu4.get()

        if not op_group or not threshold or not algorithm or not k_folds:
            messagebox.showwarning("warning", "The input field cannot be left blank.")
            return

        try:
            threshold_value = float(threshold)
            if threshold_value < 0 or threshold_value > 100:
                messagebox.showwarning("warning", "The input for the classification threshold is invalid! Please enter a positive number not greater than 100 (it is recommended to use the threshold value suggested in the drop-down box options)")
                return
        except ValueError:
            messagebox.showwarning("warning", "The format of the classification threshold is invalid! Please enter a positive number not greater than 100 (it is recommended to use the threshold value suggested in the drop-down box options)")
            return

        try:
            k_folds_value = int(k_folds)
            if k_folds_value <= 0:
                messagebox.showwarning("warning", "The number of folds for cross-validation is not valid!!! Please enter a positive integer greater than 0 (it is recommended to use the recommended number of folds from the dropdown box options)")
                return
        except ValueError:
            messagebox.showwarning("warning", "The number of folds for cross-validation is not valid!!! Please enter a positive integer greater than 0 (it is recommended to use the recommended number of folds from the dropdown box options)")
            return

        folder_path = self.process_data(op_group, threshold_value, algorithm, k_folds_value)
        self.feature_selection_and_processing(folder_path, op_group, threshold_value, algorithm, k_folds_value)
        if algorithm == "SVM":
            self.SVM_train_model(folder_path, algorithm, k_folds_value)
        elif algorithm == "Random Forest":
            self.RF_train_model(folder_path, algorithm, k_folds_value)
        elif algorithm == "KNN":
            self.KNN_train_model(folder_path, algorithm, k_folds_value)
        elif algorithm == "GBDT":
            self.GBDT_train_model(folder_path, algorithm, k_folds_value)
        elif algorithm == "SGD":
            self.SGD_train_model(folder_path, algorithm, k_folds_value)
        else:
            self.GNB_train_model(folder_path, algorithm, k_folds_value)


    def move_last_column_to_first(self, df):
        last_col = df.columns[-1]
        last_col_data = df[last_col]
        df = df.iloc[:, :-1]
        df.insert(0, last_col, last_col_data)
        return df

    def process_data(self, op_group, threshold, algorithm, k_folds):
        # Filter out all the lines that all proteins in the family can bind to  Generate the process file {file_prefix}.xlsx
        try:

            file_path = 'navicat_to_excel/Compound_OP_binding.xlsx'
            df = pd.read_excel(file_path)



            df = df[
                df['Binding Protein Name'].str.startswith(op_group)
                 & df['binding value'].notnull()
            ]

            folder_name = f"{op_group}_{threshold}_{algorithm}_{k_folds}"
            folder_path = os.path.join(
                "system_result",
                folder_name)
            os.makedirs(folder_path, exist_ok=True)

            file_prefix = "prefix"
            df.to_excel(os.path.join(folder_path, f"{file_prefix}.xlsx"), index=False)


            df = df.dropna(subset=["binding value"])


            unique_values = df["Compound name"].drop_duplicates().tolist()
            my_dict = dict.fromkeys(unique_values, None)


            for Compound_name in unique_values:
                flag = 3
                filtered_rows = df[df["Compound name"] == Compound_name]
                for _, row in filtered_rows.iterrows():
                    binding_value = row["binding value"]
                    if str(binding_value) == "nan":
                        continue
                    elif str(binding_value).startswith('>'):
                        if float(str(binding_value[1:])) >= threshold:
                            flag = 0
                            continue
                        else:
                            continue
                    else:
                        if float(str(binding_value)) <= threshold:
                            flag = 1
                            break
                        else:
                            flag = 0
                            continue
                if flag == 0:
                    my_dict[Compound_name] = "0"
                elif flag == 1:
                    my_dict[Compound_name] = "1"
                else:
                    my_dict[Compound_name] = "nan"

            my_dict = {k: v for k, v in my_dict.items() if v != "nan"}

            df_label = pd.DataFrame(list(my_dict.items()), columns=['Compound name', 'label'])
            df_label.to_excel(os.path.join(folder_path, f"{file_prefix}_label.xlsx"), index=False)

            # Write the CAS number, generate the process file {file_prefix}_cas.xlsx
            new_unique_values = list(my_dict.keys())
            cas_number_list = []
            df_cas = pd.read_excel(
                "navicat_to_excel/All_Compound_Descriptor.xlsx")
            for Compound_name in new_unique_values:
                filtered = df_cas[df_cas["Compound name"] == Compound_name]
                if not filtered.empty:
                    cas_number_list.append(filtered["CAS-number"].iloc[0])
                else:
                    cas_number_list.append("")

            df_label["CAS-number"] = cas_number_list
            df_label = self.move_last_column_to_first(df_label)
            df_label.to_excel(os.path.join(folder_path, f"{file_prefix}_label.xlsx"), index=False)

            messagebox.showinfo("Success", f"Data processing is completed and saved to the file : {folder_path}")
            return folder_path

        except Exception as e:
            messagebox.showerror(f"An error occurred when opening the file: {e}")

    def feature_selection_and_processing(self, folder_path, op_group, threshold_value, algorithm,
                                         k_folds_value):

        results_folder = os.path.join(folder_path, "results")
        os.makedirs(results_folder, exist_ok=True)
        # Read the list of compounds
        file_prefix = "prefix"
        compound_list_df = pd.read_excel(os.path.join(folder_path, f"{file_prefix}_label.xlsx"))
        compounds_to_keep = compound_list_df["Compound name"].tolist()

        # Check whether all the values in the "label" column are either 1 or 0.
        if compound_list_df['label'].nunique() == 1:
            label_value = compound_list_df['label'].unique()[0]
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Error", f"All compound labels are {label_value}，Cannot perform training. Please re-enter.")
            return

        df = pd.read_excel(
            "navicat_to_excel/All_Compound_Descriptor.xlsx")
        df = df[df["Compound name"].isin(compounds_to_keep)]
        df.to_excel(os.path.join(folder_path, "All_Compound_Descriptor_original.xlsx"), index=False)

        df_yuanshi = pd.read_excel(os.path.join(folder_path, "All_Compound_Descriptor_original.xlsx"))
        compound_list_df = pd.read_excel(os.path.join(folder_path, f"{file_prefix}_label.xlsx"))


        df_merged = df_yuanshi.merge(compound_list_df[['Compound name', 'label']], on='Compound name', how='left')


        df_merged.to_excel(os.path.join(folder_path, "All_Compound_Descriptor_original.xlsx"), index=False)

        df_yuanshi = pd.read_excel(os.path.join(folder_path, "All_Compound_Descriptor_original.xlsx"))

        label_counts = df_yuanshi['label'].value_counts()
        if any(label_counts < k_folds_value):
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Error", "The number of samples for each class must be greater than or equal to the number of folds k used in cross-validation. Please re-enter.")
            return


        smiles_to_remove = ["120068-37-3", "500008-45-7", "328263-19-0", "153783-92-7", "8016-38-4"]
        df_yuanshi = df_yuanshi[~df_yuanshi["Compound name"].isin(smiles_to_remove)]


        df_yuanshi = df_yuanshi.loc[:, (df_yuanshi != 0).any(axis=0)]
        df_yuanshi.to_excel(os.path.join(folder_path, "All_Compound_Descriptor_remove0.xlsx"), index=False)


        vif_df = pd.read_excel(
            "navicat_to_excel//Descriptor_vif_results.xlsx")
        small_dataset_df = pd.read_excel(os.path.join(folder_path, "All_Compound_Descriptor_remove0.xlsx"))
        sub_descriptor_columns = small_dataset_df.columns[2:-1].tolist()
        subset_df = vif_df[vif_df["Descriptor"].isin(sub_descriptor_columns)]
        subset_df.to_excel(os.path.join(folder_path, "vif_results.xlsx"), index=False)

        vif_df = pd.read_excel(os.path.join(folder_path, "vif_results.xlsx"))
        model_dataset = pd.read_excel(os.path.join(folder_path, "All_Compound_Descriptor_remove0.xlsx"))
        compound_descriptors = model_dataset.iloc[:, 2:-1]
        compound_descriptors = list(compound_descriptors.columns)

        if not vif_df.columns.tolist() == ["descriptor", "vif"]:


            vif_df.columns = ["descriptor", "vif"]


        compound_vif = vif_df["vif"]
        combined_dict = dict(zip(compound_descriptors, compound_vif))

        keys_to_keep = [
            key for key, value in combined_dict.items()
            if not np.isinf(value) and value > 5
        ]
        model_dataset.drop(columns=keys_to_keep, inplace=True)
        model_dataset.to_excel(os.path.join(folder_path, "All_Compound_Descriptor_remove0_removeVif.xlsx"), index=False)

        # Standard
        df_zuihou = pd.read_excel(os.path.join(folder_path, "All_Compound_Descriptor_remove0_removeVif.xlsx"))
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_zuihou.iloc[:, 2:-1])

        scaled_df = pd.DataFrame(scaled_data, columns=df_zuihou.columns[2:-1])
        output_file_scaled = os.path.join(folder_path, "All_Compound_Descriptor_remove0_removeVif_Standard.xlsx")
        scaled_df.to_excel(output_file_scaled, index=False)

        df2 = pd.read_excel(output_file_scaled)
        common_columns = df2.columns.intersection(df_zuihou.columns)
        df_zuihou[common_columns] = df2[common_columns]
        df_zuihou.to_excel(output_file_scaled, index=False)

        scaler_path = os.path.join(results_folder, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to: {scaler_path}")
        print("Scaled data has been written to", output_file_scaled)

    def SVM_train_model(self, file_path, algorithm, k_folds):
        file_path_train=os.path.join(file_path,"All_Compound_Descriptor_remove0_removeVif_Standard.xlsx")

        #training
        df = pd.read_excel(file_path_train)
        X = df.iloc[:, 2:-1]
        y = df["label"]


        y = y.replace({'active': 1, 'inactive': 0})


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        svm_model = SVC(kernel='linear', probability=True, random_state=42)


        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

        svm_model.fit(X_train,y_train)

        sfm = SelectFromModel(estimator=svm_model)
        X_train_selected = sfm.fit_transform(X_train, y_train)
        X_test_selected = sfm.transform(X_test)


        svm_model.fit(X_train_selected, y_train)


        y_pred = svm_model.predict(X_test_selected)
        y_proba = svm_model.predict_proba(X_test_selected)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        auc_roc = roc_auc_score(y_test, y_proba)
        auc_pr = average_precision_score(y_test, y_proba)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)


        print("After feature selection：")
        print("Selected Features:", X.columns[sfm.get_support()])
        print("Best Accuracy: ", accuracy)
        print("Best MCC: ", mcc)
        print("Best Recall: ", recall)
        print("Best F1 Score: ", f1)
        print("Best Precision: ", precision)
        print("AUC-ROC: ", auc_roc)
        print("AP-PR: ", auc_pr)
        print("Specificity: ", specificity)

        accuracies = []
        mccs = []
        recalls = []
        f1s = []
        precisions = []



        for train_index, test_index in skf.split(X_train_selected, y_train):
            X_train_cv, X_test_cv = X_train_selected[train_index], X_train_selected[test_index]
            y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]


            svm_model.fit(X_train_cv, y_train_cv)


            y_pred_cv = svm_model.predict(X_test_cv)
            accuracies.append(accuracy_score(y_test_cv, y_pred_cv))
            mccs.append(matthews_corrcoef(y_test_cv, y_pred_cv))
            recalls.append(recall_score(y_test_cv, y_pred_cv))
            f1s.append(f1_score(y_test_cv, y_pred_cv))
            precisions.append(precision_score(y_test_cv, y_pred_cv))



        print("---------------------------------------------------------")
        print(f"After {k_folds}-fold cross-validation:")
        print(f"Mean accuracy over {k_folds}-fold cross-validation: {np.mean(accuracies):.2f}")
        print(f"Mean MCC over {k_folds}-fold cross-validation: {np.mean(mccs):.2f}")
        print(f"Mean Recall over {k_folds}-fold cross-validation: {np.mean(recalls):.2f}")
        print(f"Mean F1 Score over {k_folds}-fold cross-validation: {np.mean(f1s):.2f}")
        print(f"Mean Precision over {k_folds}-fold cross-validation: {np.mean(precisions):.2f}")
        print("------------------------------------------------")


        param_distributions = {
            'C': loguniform(1e-3, 1e3),
            'kernel': ['linear', 'rbf'],
            'gamma': loguniform(1e-3, 1e3)
        }


        random_search = RandomizedSearchCV(estimator=svm_model, param_distributions=param_distributions,
                                           n_iter=10, cv=skf, scoring='accuracy', random_state=42)

        random_search.fit(X_train_selected, y_train)


        print("Best parameters found: ", random_search.best_params_)
        print("Best accuracy found: ", random_search.best_score_)
        print("------------------------------------------------")


        best_params = random_search.best_params_


        best_svm_model = SVC(**best_params,  probability=True, random_state=42)


        cv_accuracies = cross_val_score(best_svm_model, X_train_selected, y_train, cv=skf, scoring='accuracy')
        cv_mccs = cross_val_score(best_svm_model, X_train_selected, y_train, cv=skf, scoring='matthews_corrcoef')
        cv_recalls = cross_val_score(best_svm_model, X_train_selected, y_train, cv=skf, scoring='recall')
        cv_f1s = cross_val_score(best_svm_model, X_train_selected, y_train, cv=skf, scoring='f1')
        cv_precisions = cross_val_score(best_svm_model, X_train_selected, y_train, cv=skf, scoring='precision')
        cv_auc_rocs = cross_val_score(best_svm_model, X_train_selected, y_train, cv=skf, scoring='roc_auc')
        cv_ap_prs = cross_val_score(best_svm_model, X_train_selected, y_train, cv=skf, scoring='average_precision')

        from sklearn.metrics import make_scorer

        def specificity_score(y_true, y_pred):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()


            return tn / (tn + fp)

        specificity_scorer = make_scorer(specificity_score)

        cv_specificities = cross_val_score(best_svm_model, X_train_selected, y_train, cv=skf,
                                           scoring=specificity_scorer)

        print("Best parameters found: ", best_params)
        print("Best Accuracy: ", np.mean(cv_accuracies))
        print("Best MCC: ", np.mean(cv_mccs))
        print("Best Recall: ", np.mean(cv_recalls))
        print("Best F1 Score: ", np.mean(cv_f1s))
        print("Best Precision: ", np.mean(cv_precisions))
        print("Best AUC-ROC: ", np.mean(cv_auc_rocs))
        print("Best AP-PR ", np.mean(cv_ap_prs))
        print("Best Specificity: ", np.mean(cv_specificities))

        best_svm_model.fit(X_train_selected, y_train)
        y_pred_proba = best_svm_model.predict_proba(X_test_selected)[:, 1]
        auc_roc_best = roc_auc_score(y_test, y_pred_proba)
        auc_pr_best = average_precision_score(y_test, y_pred_proba)

        y_pred_best = best_svm_model.predict(X_test_selected)
        tn_best, fp_best, fn_best, tp_best = confusion_matrix(y_test, y_pred_best).ravel()
        specificity_best = tn_best / (tn_best + fp_best)

        results_dir = os.path.join(file_path, "results")
        os.makedirs(results_dir, exist_ok=True)

        best_svm_model.fit(X_train_selected, y_train)
        y_pred_best = best_svm_model.predict(X_test_selected)
        y_pred_best_proba = best_svm_model.predict_proba(X_test_selected)[:, 1]


        best_accuracy = accuracy_score(y_test, y_pred_best)
        best_mcc = matthews_corrcoef(y_test, y_pred_best)
        best_recall = recall_score(y_test, y_pred_best)
        best_f1 = f1_score(y_test, y_pred_best)
        best_precision = precision_score(y_test, y_pred_best)


        precision_pr_best, recall_pr_best, _ = precision_recall_curve(y_test, y_pred_best_proba)
        pr_auc_best = auc(recall_pr_best, precision_pr_best)


        fpr_best, tpr_best, _ = roc_curve(y_test, y_pred_best_proba)
        roc_auc_best = auc(fpr_best, tpr_best)


        cm_best = confusion_matrix(y_test, y_pred_best)


        plt.figure()
        sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('SVM')
        best_cm_path = os.path.join(results_dir, 'Confusion_Matrix.png')
        plt.savefig(best_cm_path)

        plt.clf()


        plt.figure()
        plt.plot(recall_pr_best, precision_pr_best, label=f'PR curve (area = {pr_auc_best:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('SVM——PR')
        plt.legend(loc="lower right")
        best_pr_curve_path = os.path.join(results_dir, 'PR_curve.png')
        plt.savefig(best_pr_curve_path)

        plt.clf()


        plt.figure()
        plt.plot(fpr_best, tpr_best, label=f'ROC curve (area = {roc_auc_best:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('SVM——ROC')
        plt.legend(loc="lower right")
        best_roc_curve_path = os.path.join(results_dir, 'ROC_curve.png')
        plt.savefig(best_roc_curve_path)

        plt.clf()



        results = {
            "Feature selection：": {
                "Accuracy": accuracy,
                "MCC": mcc,
                "Recall": recall,
                "F1 Score": f1,
                "Precision": precision,
                "AUC-ROC": auc_roc,
                "AP-PR": auc_pr,
                "Specificity": specificity

            },
            "Feature selection + cross-validation：": {
                "Mean Accuracy": np.mean(accuracies),
                "Mean MCC": np.mean(mccs),
                "Mean Recall": np.mean(recalls),
                "Mean F1 Score": np.mean(f1s),
                "Mean Precision": np.mean(precisions),

            },
            "Feature selection + cross-validation + random search：": {
                "Best parameters found": best_params,
                "Mean Accuracy": np.mean(cv_accuracies),
                "Mean MCC": np.mean(cv_mccs),
                "Mean Recall": np.mean(cv_recalls),
                "Mean F1 Score": np.mean(cv_f1s),
                "Mean Precision": np.mean(cv_precisions),
                "Mean AUC-ROC": np.mean(cv_auc_rocs),
                "Mean AP-PR": np.mean(cv_ap_prs),
                "Mean Specificity": np.mean(cv_specificities)
            }
        }



        results_file = os.path.join(results_dir, "results.json")
        with open(results_file, "w",encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        messagebox.showinfo("Training results", f"The result has been saved to the file.: {results_file}")

        # SHAP Explainable

        best_svm_model.fit(X_train_selected, y_train)
        best_svm_model.fit(X_train_selected, y_train)

        try:
            explainer = shap.Explainer(best_svm_model, X_train_selected)
            shap_values = explainer.shap_values(X_test_selected)
        except TypeError as e:
            print(f"An error occurred while using shap.Explainer: {e}")
            print("Try to use KernelExplainer instead...")
            explainer = shap.KernelExplainer(best_svm_model.predict, X_train_selected)
            shap_values = explainer.shap_values(X_test_selected, nsamples=100)
        except np.linalg.LinAlgError as e:
            print("Error with KDE: ", e)
            print("Please try other algorithms.")

        # save
        results_dir = os.path.join(file_path, "results")
        os.makedirs(results_dir, exist_ok=True)
        # Display the specific descriptor name
        selected_feature_names = X.columns[sfm.get_support()]


        if isinstance(X_test_selected, np.ndarray):
            X_test_selected = pd.DataFrame(X_test_selected, columns=selected_feature_names)
        try:
          # SHAP Summary Plot (Violin Plot)
            shap.summary_plot(shap_values, X_test_selected, feature_names=selected_feature_names, plot_type="violin",
                              show=False)
            plt.title('SHAP Summary Plot (Violin)')
            plt.xlabel('SHAP Value')
            plt.ylabel('Features')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'SHAP_Summary_Violin_Plot.png'))
            # plt.show()

            plt.clf()

            # SHAP Bar Plot
            shap.summary_plot(shap_values, X_test_selected, feature_names=selected_feature_names, plot_type="bar",
                              show=False)
            plt.title('SHAP Summary Plot (Bar)')
            plt.xlabel('Mean Absolute SHAP Value')
            plt.ylabel('Features')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'SHAP_Summary_Bar_Plot.png'))
            # plt.show()

            plt.clf()


            shap_values_df = pd.DataFrame(shap_values, columns=selected_feature_names)
            plt.figure(figsize=(12, 8))
            sns.heatmap(shap_values_df, cmap='coolwarm', center=0)
            plt.title('SHAP Values Heatmap')
            plt.xlabel('Features')
            plt.ylabel('Samples')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'SHAP_Heatmap.png'))
            # plt.show()

            plt.clf()
        except np.linalg.LinAlgError as e:
            print("Error with KDE: ", e)
            print("Please try other algorithms.")

        messagebox.showinfo("The SHAP results have been explained and the results have been saved to a file: ", results_dir)

        # t-SNE

        data = pd.read_excel(file_path_train, None)

        # 1、Data after feature selection: t-SNE dimensionality reduction visualization
        tsne_perplexity = min(30, X_train_selected.shape[0] - 1)  # Make sure that the perplexity is less than the sample size.
        tsne = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity)
        X_tsne = tsne.fit_transform(X_train_selected)


        X_tsne_data = pd.DataFrame({
            'Dim1': X_tsne[:, 0],
            'Dim2': X_tsne[:, 1],
            'label': y_train
        })

        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=X_tsne_data, x='Dim1', y='Dim2', hue='label', palette='viridis', alpha=0.7, edgecolor='k',
                        s=100)

        plt.title("t-SNE Visualization of Selected Features", fontsize=16)
        plt.xlabel("Dimension 1", fontsize=14)
        plt.ylabel("Dimension 2", fontsize=14)


        legend = plt.legend(title='Class Label', fontsize=12, title_fontsize=14, loc='best', borderpad=1)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1.5)


        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)


        tsne_plot_path = os.path.join(results_dir, "tsne_plot_selected.png")
        plt.savefig(tsne_plot_path, dpi=300, bbox_inches='tight')

        plt.clf()

        # plt.show()


        data = pd.read_excel(file_path_train, None)


        for sh_name, sh_data in data.items():

            X = sh_data.drop(['Compound name', 'CAS-number', 'label'], axis=1)
            y_true = sh_data['label']


            y_true_str = y_true.map({0: 'y_true_inactive', 1: 'y_true_active'})


            cat_cols = X.columns.tolist()
            for col in cat_cols:
                X[col] = X[col].astype("category")
                X[col] = X[col].cat.codes


            X_std = StandardScaler().fit_transform(X)


            tsne_perplexity = min(30, X_std.shape[0] - 1)
            tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=42)
            X_tsne = tsne.fit_transform(X_std)

            svm_model = SVC(kernel='linear', random_state=42)
            svm_model.fit(X_std, y_true)
            y_pred = svm_model.predict(X_std)


            y_pred_str = pd.Series(y_pred).map({0: 'y_pred_inactive', 1: 'y_pred_active'})


            X_tsne_data = pd.DataFrame({
                'Dim1': X_tsne[:, 0],
                'Dim2': X_tsne[:, 1],
                'true_label': y_true_str,
                'pred_label': y_pred_str,
                'combined_label': y_true_str + ' / ' + y_pred_str
            })


            plt.figure(figsize=(12, 10))
            ax = sns.scatterplot(data=X_tsne_data, hue='combined_label', x='Dim1', y='Dim2',
                                 palette={
                                     'y_true_inactive / y_pred_inactive': 'blue',
                                     'y_true_inactive / y_pred_active': 'purple',
                                     'y_true_active / y_pred_inactive': 'orange',
                                     'y_true_active / y_pred_active': 'green'
                                 }, alpha=0.7, edgecolor='k', s=100)


            plt.title("t-SNE Visualization of Complete Database", fontsize=16)
            plt.xlabel("Dimension 1", fontsize=14)
            plt.ylabel("Dimension 2", fontsize=14)


            legend = plt.legend(title='Class Label', fontsize=12, title_fontsize=14, loc='best', borderpad=1)
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(1.5)

            plt.grid(True, linestyle='--', linewidth=0.5)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)


            tsne_plot_path_all = os.path.join(results_dir, f"tsne_plot_all.png")
            plt.savefig(tsne_plot_path_all, dpi=300, bbox_inches='tight')
            # plt.show()

            plt.clf()
        messagebox.showinfo("The t-SNE visualization results have been saved to a file: ",results_dir)

      # save model
        try:

            feature_selection_file = os.path.join(results_dir, "selected_features.xlsx")
            pd.DataFrame(selected_feature_names, columns=["Selected Features"]).to_excel(feature_selection_file,
                                                                                        index=False)
            user_response = messagebox.askokcancel("save model", "Do you want to save the model file?")
            if user_response:

                model_dir = os.path.join(file_path, "train_model")
                os.makedirs(model_dir, exist_ok=True)
                model_filePath = os.path.join(model_dir, "model.pkl")
                with open(model_filePath, "wb") as f:
                    pickle.dump(best_svm_model, f)
                messagebox.showinfo("message", "Model has been saved")
            else:

                messagebox.showinfo("message", "Operation has been cancelled.")

        except Exception as e:
            messagebox.showerror("error", f"An error occurred during the model saving process. {e}")


    def KNN_train_model(self, file_path, algorithm, k_folds):
        file_path_train = os.path.join(file_path, "All_Compound_Descriptor_remove0_removeVif_Standard.xlsx")


        df = pd.read_excel(file_path_train)
        X = df.iloc[:, 2:-1]
        y = df["label"]


        y = y.replace({'active': 1, 'inactive': 0})


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        knn_model = KNeighborsClassifier()


        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

        skb = SelectKBest(score_func=f_classif, k='all')
        X_train_selected = skb.fit_transform(X_train, y_train)
        X_test_selected = skb.transform(X_test)


        knn_model.fit(X_train_selected, y_train)


        y_pred = knn_model.predict(X_test_selected)
        y_proba = knn_model.predict_proba(X_test_selected)[:, 1]


        accuracy = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        auc_roc = roc_auc_score(y_test, y_proba)
        auc_pr = average_precision_score(y_test, y_proba)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)


        print("After feature selection:")
        print("Selected Features:", X.columns[skb.get_support()])
        print("Best Accuracy: ", accuracy)
        print("Best MCC: ", mcc)
        print("Best Recall: ", recall)
        print("Best F1 Score: ", f1)
        print("Best Precision: ", precision)
        print("AUC-ROC: ", auc_roc)
        print("AP-PR: ", auc_pr)
        print("Specificity: ", specificity)


        accuracies = []
        mccs = []
        recalls = []
        f1s = []
        precisions = []


        for train_index, test_index in skf.split(X_train_selected, y_train):
            X_train_cv, X_test_cv = X_train_selected[train_index], X_train_selected[test_index]
            y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]


            knn_model.fit(X_train_cv, y_train_cv)


            y_pred_cv = knn_model.predict(X_test_cv)
            accuracies.append(accuracy_score(y_test_cv, y_pred_cv))
            mccs.append(matthews_corrcoef(y_test_cv, y_pred_cv))
            recalls.append(recall_score(y_test_cv, y_pred_cv))
            f1s.append(f1_score(y_test_cv, y_pred_cv))
            precisions.append(precision_score(y_test_cv, y_pred_cv))


        print("---------------------------------------------------------")
        print(f"After {k_folds}-fold cross-validation:")
        print(f"Mean accuracy over {k_folds}-fold cross-validation: {np.mean(accuracies):.2f}")
        print(f"Mean MCC over {k_folds}-fold cross-validation: {np.mean(mccs):.2f}")
        print(f"Mean Recall over {k_folds}-fold cross-validation: {np.mean(recalls):.2f}")
        print(f"Mean F1 Score over {k_folds}-fold cross-validation: {np.mean(f1s):.2f}")
        print(f"Mean Precision over {k_folds}-fold cross-validation: {np.mean(precisions):.2f}")
        print("------------------------------------------------")


        param_distributions = {
            'n_neighbors': randint(1, 50),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }


        random_search = RandomizedSearchCV(estimator=knn_model, param_distributions=param_distributions,
                                           n_iter=10, cv=skf, scoring='accuracy', random_state=42)


        random_search.fit(X_train_selected, y_train)


        print("Best parameters found: ", random_search.best_params_)
        print("Best accuracy found: ", random_search.best_score_)
        print("------------------------------------------------")


        best_params = random_search.best_params_


        best_knn_model = KNeighborsClassifier(**best_params)


        cv_accuracies = cross_val_score(best_knn_model, X_train_selected, y_train, cv=skf, scoring='accuracy')
        cv_mccs = cross_val_score(best_knn_model, X_train_selected, y_train, cv=skf, scoring='matthews_corrcoef')
        cv_recalls = cross_val_score(best_knn_model, X_train_selected, y_train, cv=skf, scoring='recall')
        cv_f1s = cross_val_score(best_knn_model, X_train_selected, y_train, cv=skf, scoring='f1')
        cv_precisions = cross_val_score(best_knn_model, X_train_selected, y_train, cv=skf, scoring='precision')
        cv_auc_rocs = cross_val_score(best_knn_model, X_train_selected, y_train, cv=skf, scoring='roc_auc')
        cv_ap_prs = cross_val_score(best_knn_model, X_train_selected, y_train, cv=skf, scoring='average_precision')


        from sklearn.metrics import make_scorer

        def specificity_score(y_true, y_pred):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tn / (tn + fp)

        specificity_scorer = make_scorer(specificity_score)

        cv_specificities = cross_val_score(best_knn_model, X_train_selected, y_train, cv=skf,
                                           scoring=specificity_scorer)

        print("Best parameters found: ", best_params)
        print("Best Accuracy: ", np.mean(cv_accuracies))
        print("Best MCC: ", np.mean(cv_mccs))
        print("Best Recall: ", np.mean(cv_recalls))
        print("Best F1 Score: ", np.mean(cv_f1s))
        print("Best Precision: ", np.mean(cv_precisions))
        print("Best AUC-ROC: ", np.mean(cv_auc_rocs))
        print("Best AP-PR ", np.mean(cv_ap_prs))
        print("Best Specificity: ", np.mean(cv_specificities))


        best_knn_model.fit(X_train_selected, y_train)
        y_pred_proba = best_knn_model.predict_proba(X_test_selected)[:, 1]
        auc_roc_best = roc_auc_score(y_test, y_pred_proba)
        auc_pr_best = average_precision_score(y_test, y_pred_proba)


        y_pred_best = best_knn_model.predict(X_test_selected)
        tn_best, fp_best, fn_best, tp_best = confusion_matrix(y_test, y_pred_best).ravel()
        specificity_best = tn_best / (tn_best + fp_best)


        results_dir = os.path.join(file_path, "results")
        os.makedirs(results_dir, exist_ok=True)

        best_knn_model.fit(X_train_selected, y_train)
        y_pred_best = best_knn_model.predict(X_test_selected)
        y_pred_best_proba = best_knn_model.predict_proba(X_test_selected)[:, 1]


        best_accuracy = accuracy_score(y_test, y_pred_best)
        best_mcc = matthews_corrcoef(y_test, y_pred_best)
        best_recall = recall_score(y_test, y_pred_best)
        best_f1 = f1_score(y_test, y_pred_best)
        best_precision = precision_score(y_test, y_pred_best)



        precision_pr_best, recall_pr_best, _ = precision_recall_curve(y_test, y_pred_best_proba)
        pr_auc_best = auc(recall_pr_best, precision_pr_best)

        fpr_best, tpr_best, _ = roc_curve(y_test, y_pred_best_proba)
        roc_auc_best = auc(fpr_best, tpr_best)


        cm_best = confusion_matrix(y_test, y_pred_best)

        plt.figure()
        sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('KNN')
        best_cm_path = os.path.join(results_dir, 'Confusion_Matrix.png')
        plt.savefig(best_cm_path)

        plt.clf()


        plt.figure()
        plt.plot(recall_pr_best, precision_pr_best, label=f'PR curve (area = {pr_auc_best:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('KNN——PR')
        plt.legend(loc="lower right")
        best_pr_curve_path = os.path.join(results_dir, 'PR_curve.png')
        plt.savefig(best_pr_curve_path)

        plt.clf()


        plt.figure()
        plt.plot(fpr_best, tpr_best, label=f'ROC curve (area = {roc_auc_best:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('KNN——ROC')
        plt.legend(loc="lower right")
        best_roc_curve_path = os.path.join(results_dir, 'ROC_curve.png')
        plt.savefig(best_roc_curve_path)

        plt.clf()


        results = {
            "Feature selection:": {
                "Accuracy": accuracy,
                "MCC": mcc,
                "Recall": recall,
                "F1 Score": f1,
                "Precision": precision,
                "AUC-ROC": auc_roc,
                "AP-PR": auc_pr,
                "Specificity": specificity

            },
            "Feature selection + cross-validation": {
                "Mean Accuracy": np.mean(accuracies),
                "Mean MCC": np.mean(mccs),
                "Mean Recall": np.mean(recalls),
                "Mean F1 Score": np.mean(f1s),
                "Mean Precision": np.mean(precisions),
            },
            "Feature selection + cross-validation + random search：": {
                "Best parameters found": best_params,
                "Mean Accuracy": np.mean(cv_accuracies),
                "Mean MCC": np.mean(cv_mccs),
                "Mean Recall": np.mean(cv_recalls),
                "Mean F1 Score": np.mean(cv_f1s),
                "Mean Precision": np.mean(cv_precisions),
                "Mean AUC-ROC": np.mean(cv_auc_rocs),
                "Mean AP-PR": np.mean(cv_ap_prs),
                "Mean Specificity": np.mean(cv_specificities)
            }

        }

        results_file = os.path.join(results_dir, "results.json")
        with open(results_file, "w", encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        messagebox.showinfo("Training results", f"The result has been saved to the file.: {results_file}")

        # # lime
        explainer = LimeTabularExplainer(X_train_selected,
                                         mode="classification",
                                         feature_names=X.columns[skb.get_support()].tolist(),
                                         discretize_continuous=True)


        idx = random.randint(0, len(X_test) - 1)


        sample_data = X_test.iloc[idx]


        sample_data_array = sample_data.values


        exp = explainer.explain_instance(sample_data_array, knn_model.predict_proba, num_features=50, top_labels=2,
                                         num_samples=1000)
        exp.save_to_file(os.path.join(results_dir,'lime_explanation.html'))


        lime_results = {
            "instance_index": str(idx),
            "predicted_class": int(knn_model.predict(X_test_selected[idx].reshape(1, -1))[0]),
            "lime_explanation": {
                "as_list": exp.as_list(),
                "as_map": exp.as_map(),
                "predict_proba": knn_model.predict_proba([X_test_selected[idx]]).tolist()
            }
        }


        lime_results_file = os.path.join(results_dir, "lime_explanation.txt")
        with open(lime_results_file, "w", encoding="utf-8") as f:
            f.write("Lime result\n\n")
            f.write(f"Example Index: {lime_results['instance_index']}\n")
            f.write(f"Prediction category: {lime_results['predicted_class']}\n\n")

            f.write("Lime result:\n")
            f.write("As List:\n")
            for feature, weight in lime_results['lime_explanation']['as_list']:
                f.write(f"{feature}: {weight}\n")
            f.write("\nAs Map:\n")
            for label, weights in lime_results['lime_explanation']['as_map'].items():
                f.write(f"Label: {label}\n")
                for feature, weight in weights:
                    f.write(f"{feature}: {weight}\n")
            f.write("\nPredicted Probabilities:\n")
            for prob in lime_results['lime_explanation']['predict_proba']:
                f.write(f"{prob}\n")

        messagebox.showinfo("Lime can interpret the results. The results have been saved to a file: ",results_dir)

        # t-SNE


        data = pd.read_excel(file_path_train, None)


        tsne_perplexity = min(30, X_train_selected.shape[0] - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity)
        X_tsne = tsne.fit_transform(X_train_selected)


        X_tsne_data = pd.DataFrame({
            'Dim1': X_tsne[:, 0],
            'Dim2': X_tsne[:, 1],
            'label': y_train
        })


        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=X_tsne_data, x='Dim1', y='Dim2', hue='label', palette='viridis', alpha=0.7, edgecolor='k',
                        s=100)


        plt.title("t-SNE Visualization of Selected Features", fontsize=16)
        plt.xlabel("Dimension 1", fontsize=14)
        plt.ylabel("Dimension 2", fontsize=14)

        legend = plt.legend(title='Class Label', fontsize=12, title_fontsize=14, loc='best', borderpad=1)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1.5)

        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)


        tsne_plot_path = os.path.join(results_dir, "tsne_plot_selected.png")
        plt.savefig(tsne_plot_path, dpi=300, bbox_inches='tight')

        plt.clf()





        data = pd.read_excel(file_path_train, None)


        for sh_name, sh_data in data.items():

            X = sh_data.drop(['Compound name', 'CAS-number', 'label'], axis=1)
            y_true = sh_data['label']

            y_true_str = y_true.map({0: 'y_true_inactive', 1: 'y_true_active'})


            cat_cols = X.columns.tolist()
            for col in cat_cols:
                X[col] = X[col].astype("category")
                X[col] = X[col].cat.codes


            X_std = StandardScaler().fit_transform(X)

            tsne_perplexity = min(30, X_std.shape[0] - 1)
            tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=42)
            X_tsne = tsne.fit_transform(X_std)


            knn_model = KNeighborsClassifier()
            knn_model.fit(X_std, y_true)
            y_pred = knn_model.predict(X_std)


            y_pred_str = pd.Series(y_pred).map({0: 'y_pred_inactive', 1: 'y_pred_active'})


            X_tsne_data = pd.DataFrame({
                'Dim1': X_tsne[:, 0],
                'Dim2': X_tsne[:, 1],
                'true_label': y_true_str,
                'pred_label': y_pred_str,
                'combined_label': y_true_str + ' / ' + y_pred_str
            })


            plt.figure(figsize=(12, 10))
            ax = sns.scatterplot(data=X_tsne_data, hue='combined_label', x='Dim1', y='Dim2',
                                 palette={
                                     'y_true_inactive / y_pred_inactive': 'blue',
                                     'y_true_inactive / y_pred_active': 'purple',
                                     'y_true_active / y_pred_inactive': 'orange',
                                     'y_true_active / y_pred_active': 'green'
                                 }, alpha=0.7, edgecolor='k', s=100)


            plt.title("t-SNE Visualization of Complete Database", fontsize=16)
            plt.xlabel("Dimension 1", fontsize=14)
            plt.ylabel("Dimension 2", fontsize=14)


            legend = plt.legend(title='Class Label', fontsize=12, title_fontsize=14, loc='best', borderpad=1)
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(1.5)


            plt.grid(True, linestyle='--', linewidth=0.5)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            tsne_plot_path_all = os.path.join(results_dir, f"tsne_plot_all.png")
            plt.savefig(tsne_plot_path_all, dpi=300, bbox_inches='tight')
            # plt.show()

            plt.clf()
        messagebox.showinfo("The t-SNE visualization results have been saved to a file: ", results_dir)


        try:

            selected_feature_names = X.columns[skb.get_support()]
            feature_selection_file = os.path.join(results_dir, "selected_features.xlsx")
            pd.DataFrame(selected_feature_names, columns=["Selected Features"]).to_excel(feature_selection_file,
                                                                                         index=False)
            user_response = messagebox.askokcancel("save model", "Do you want to save the model file?")
            if user_response:

                model_dir = os.path.join(file_path, "train_model")
                os.makedirs(model_dir, exist_ok=True)
                model_filePath = os.path.join(model_dir, "model.pkl")
                with open(model_filePath, "wb") as f:
                    pickle.dump(best_knn_model, f)
                messagebox.showinfo("message", "Model has been saved")
            else:

                messagebox.showinfo("message", "Operation has been cancelled.")

        except Exception as e:
            messagebox.showerror("error", f"An error occurred during the model saving process: {e}")

    def RF_train_model(self, file_path, algorithm, k_folds):
        file_path_train = os.path.join(file_path, "All_Compound_Descriptor_remove0_removeVif_Standard.xlsx")


        df = pd.read_excel(file_path_train)
        X = df.iloc[:, 2:-1]
        y = df["label"]


        y = y.replace({'active': 1, 'inactive': 0})

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        rf_model = RandomForestClassifier(random_state=42)

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

        rf_model.fit(X_train, y_train)

        sfm = SelectFromModel(estimator=rf_model)
        X_train_selected = sfm.fit_transform(X_train, y_train)
        X_test_selected = sfm.transform(X_test)


        rf_model.fit(X_train_selected, y_train)


        y_pred = rf_model.predict(X_test_selected)
        y_proba = rf_model.predict_proba(X_test_selected)[:, 1]


        accuracy = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        auc_roc = roc_auc_score(y_test, y_proba)
        auc_pr = average_precision_score(y_test, y_proba)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)



        print("Selected Features:", X.columns[sfm.get_support()])
        print("Best Accuracy: ", accuracy)
        print("Best MCC: ", mcc)
        print("Best Recall: ", recall)
        print("Best F1 Score: ", f1)
        print("Best Precision: ", precision)
        print("AUC-ROC: ", auc_roc)
        print("AP-PR: ", auc_pr)
        print("Specificity: ", specificity)


        accuracies = []
        mccs = []
        recalls = []
        f1s = []
        precisions = []


        for train_index, test_index in skf.split(X_train_selected, y_train):
            X_train_cv, X_test_cv = X_train_selected[train_index], X_train_selected[test_index]
            y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]


            rf_model.fit(X_train_cv, y_train_cv)


            y_pred_cv = rf_model.predict(X_test_cv)
            accuracies.append(accuracy_score(y_test_cv, y_pred_cv))
            mccs.append(matthews_corrcoef(y_test_cv, y_pred_cv))
            recalls.append(recall_score(y_test_cv, y_pred_cv))
            f1s.append(f1_score(y_test_cv, y_pred_cv))
            precisions.append(precision_score(y_test_cv, y_pred_cv))



        print("---------------------------------------------------------")
        print(f"After {k_folds}-fold cross-validation:")
        print(f"Mean accuracy over {k_folds}-fold cross-validation: {np.mean(accuracies):.2f}")
        print(f"Mean MCC over {k_folds}-fold cross-validation: {np.mean(mccs):.2f}")
        print(f"Mean Recall over {k_folds}-fold cross-validation: {np.mean(recalls):.2f}")
        print(f"Mean F1 Score over {k_folds}-fold cross-validation: {np.mean(f1s):.2f}")
        print(f"Mean Precision over {k_folds}-fold cross-validation: {np.mean(precisions):.2f}")
        print("------------------------------------------------")

        param_distributions = {
            'n_estimators': [50, 100, 200],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }


        random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_distributions,
                                           n_iter=10, cv=skf, scoring='accuracy', random_state=42)

        random_search.fit(X_train_selected, y_train)


        print("Best parameters found: ", random_search.best_params_)
        print("Best accuracy found: ", random_search.best_score_)
        print("------------------------------------------------")

        best_params = random_search.best_params_


        best_rf_model = RandomForestClassifier(**best_params, random_state=42)

        cv_accuracies = cross_val_score(best_rf_model, X_train_selected, y_train, cv=skf, scoring='accuracy')
        cv_mccs = cross_val_score(best_rf_model, X_train_selected, y_train, cv=skf, scoring='matthews_corrcoef')
        cv_recalls = cross_val_score(best_rf_model, X_train_selected, y_train, cv=skf, scoring='recall')
        cv_f1s = cross_val_score(best_rf_model, X_train_selected, y_train, cv=skf, scoring='f1')
        cv_precisions = cross_val_score(best_rf_model, X_train_selected, y_train, cv=skf, scoring='precision')
        cv_auc_rocs = cross_val_score(best_rf_model, X_train_selected, y_train, cv=skf, scoring='roc_auc')
        cv_ap_prs = cross_val_score(best_rf_model, X_train_selected, y_train, cv=skf, scoring='average_precision')


        from sklearn.metrics import make_scorer

        def specificity_score(y_true, y_pred):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tn / (tn + fp)

        specificity_scorer = make_scorer(specificity_score)

        cv_specificities = cross_val_score(best_rf_model, X_train_selected, y_train, cv=skf,
                                           scoring=specificity_scorer)

        print("Best parameters found: ", best_params)
        print("Best Accuracy: ", np.mean(cv_accuracies))
        print("Best MCC: ", np.mean(cv_mccs))
        print("Best Recall: ", np.mean(cv_recalls))
        print("Best F1 Score: ", np.mean(cv_f1s))
        print("Best Precision: ", np.mean(cv_precisions))
        print("Best AUC-ROC: ", np.mean(cv_auc_rocs))
        print("Best AP-PR ", np.mean(cv_ap_prs))
        print("Best Specificity: ", np.mean(cv_specificities))

        best_rf_model.fit(X_train_selected, y_train)
        y_pred_proba = best_rf_model.predict_proba(X_test_selected)[:, 1]
        auc_roc_best = roc_auc_score(y_test, y_pred_proba)
        auc_pr_best = average_precision_score(y_test, y_pred_proba)


        y_pred_best = best_rf_model.predict(X_test_selected)
        tn_best, fp_best, fn_best, tp_best = confusion_matrix(y_test, y_pred_best).ravel()
        specificity_best = tn_best / (tn_best + fp_best)


        results_dir = os.path.join(file_path, "results")
        os.makedirs(results_dir, exist_ok=True)

        best_rf_model.fit(X_train_selected, y_train)
        y_pred_best = best_rf_model.predict(X_test_selected)
        y_pred_best_proba = best_rf_model.predict_proba(X_test_selected)[:, 1]

        best_accuracy = accuracy_score(y_test, y_pred_best)
        best_mcc = matthews_corrcoef(y_test, y_pred_best)
        best_recall = recall_score(y_test, y_pred_best)
        best_f1 = f1_score(y_test, y_pred_best)
        best_precision = precision_score(y_test, y_pred_best)

        precision_pr_best, recall_pr_best, _ = precision_recall_curve(y_test, y_pred_best_proba)
        pr_auc_best = auc(recall_pr_best, precision_pr_best)

        fpr_best, tpr_best, _ = roc_curve(y_test, y_pred_best_proba)
        roc_auc_best = auc(fpr_best, tpr_best)

        cm_best = confusion_matrix(y_test, y_pred_best)

        plt.figure()
        sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('RF')
        best_cm_path = os.path.join(results_dir, 'Confusion_Matrix.png')
        plt.savefig(best_cm_path)

        plt.clf()

        plt.figure()
        plt.plot(recall_pr_best, precision_pr_best, label=f'PR curve (area = {pr_auc_best:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('RF——PR')
        plt.legend(loc="lower right")
        best_pr_curve_path = os.path.join(results_dir, 'PR_curve.png')
        plt.savefig(best_pr_curve_path)

        plt.clf()


        plt.figure()
        plt.plot(fpr_best, tpr_best, label=f'ROC curve (area = {roc_auc_best:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('RF——ROC')
        plt.legend(loc="lower right")
        best_roc_curve_path = os.path.join(results_dir, 'ROC_curve.png')
        plt.savefig(best_roc_curve_path)

        plt.clf()



        results = {
            "Feature selection:": {
                "Accuracy": accuracy,
                "MCC": mcc,
                "Recall": recall,
                "F1 Score": f1,
                "Precision": precision,
                "AUC-ROC": auc_roc,
                "AP-PR": auc_pr,
                "Specificity": specificity

            },
            "Feature selection + cross-validation": {
                "Mean Accuracy": np.mean(accuracies),
                "Mean MCC": np.mean(mccs),
                "Mean Recall": np.mean(recalls),
                "Mean F1 Score": np.mean(f1s),
                "Mean Precision": np.mean(precisions),
            },
            "Feature selection + cross-validation + random search：": {
                "Best parameters found": best_params,
                "Mean Accuracy": np.mean(cv_accuracies),
                "Mean MCC": np.mean(cv_mccs),
                "Mean Recall": np.mean(cv_recalls),
                "Mean F1 Score": np.mean(cv_f1s),
                "Mean Precision": np.mean(cv_precisions),
                "Mean AUC-ROC": np.mean(cv_auc_rocs),
                "Mean AP-PR": np.mean(cv_ap_prs),
                "Mean Specificity": np.mean(cv_specificities)
            }

        }

        results_file = os.path.join(results_dir, "results.json")
        with open(results_file, "w", encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        messagebox.showinfo("Training conclusion", f"The result has been saved to the file: {results_file}")


        best_rf_model.fit(X_train_selected, y_train)
        explainer = lime.lime_tabular.LimeTabularExplainer(X_train_selected, mode='classification',
                                                           feature_names=X.columns[sfm.get_support()].tolist())


        idx = random.randint(0, len(X_test_selected) - 1)


        if isinstance(X_test_selected, np.ndarray):
            sample_data = pd.DataFrame(X_test_selected[idx].reshape(1, -1), columns=X.columns[sfm.get_support()])
        else:
            sample_data = X_test_selected.iloc[[idx]]


        sample_data_array = sample_data.values.reshape(1, -1)


        explanation = explainer.explain_instance(sample_data_array[0], best_rf_model.predict_proba, num_features=50,
                                                 top_labels=2,
                                                 num_samples=1000)
        explanation.save_to_file(os.path.join(results_dir,'lime_explanation_randomForest.html'))
        messagebox.showinfo("Lime can interpret the results. The results have been saved to a file: ", results_dir)

        # t-SNE


        data = pd.read_excel(file_path_train, None)


        tsne_perplexity = min(30, X_train_selected.shape[0] - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity)
        X_tsne = tsne.fit_transform(X_train_selected)


        X_tsne_data = pd.DataFrame({
            'Dim1': X_tsne[:, 0],
            'Dim2': X_tsne[:, 1],
            'label': y_train
        })


        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=X_tsne_data, x='Dim1', y='Dim2', hue='label', palette='viridis', alpha=0.7, edgecolor='k',
                        s=100)


        plt.title("t-SNE Visualization of Selected Features", fontsize=16)
        plt.xlabel("Dimension 1", fontsize=14)
        plt.ylabel("Dimension 2", fontsize=14)


        legend = plt.legend(title='Class Label', fontsize=12, title_fontsize=14, loc='best', borderpad=1)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1.5)


        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)


        tsne_plot_path = os.path.join(results_dir, "tsne_plot_selected.png")
        plt.savefig(tsne_plot_path, dpi=300, bbox_inches='tight')

        plt.clf()

        # plt.show()



        data = pd.read_excel(file_path_train, None)


        for sh_name, sh_data in data.items():

            X = sh_data.drop(['Compound name', 'CAS-number', 'label'], axis=1)
            y_true = sh_data['label']


            y_true_str = y_true.map({0: 'y_true_inactive', 1: 'y_true_active'})

            cat_cols = X.columns.tolist()
            for col in cat_cols:
                X[col] = X[col].astype("category")
                X[col] = X[col].cat.codes


            X_std = StandardScaler().fit_transform(X)


            tsne_perplexity = min(30, X_std.shape[0] - 1)
            tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=42)
            X_tsne = tsne.fit_transform(X_std)


            rf_model = RandomForestClassifier()
            rf_model.fit(X_std, y_true)
            y_pred = rf_model.predict(X_std)


            y_pred_str = pd.Series(y_pred).map({0: 'y_pred_inactive', 1: 'y_pred_active'})


            X_tsne_data = pd.DataFrame({
                'Dim1': X_tsne[:, 0],
                'Dim2': X_tsne[:, 1],
                'true_label': y_true_str,
                'pred_label': y_pred_str,
                'combined_label': y_true_str + ' / ' + y_pred_str
            })


            plt.figure(figsize=(12, 10))
            ax = sns.scatterplot(data=X_tsne_data, hue='combined_label', x='Dim1', y='Dim2',
                                 palette={
                                     'y_true_inactive / y_pred_inactive': 'blue',
                                     'y_true_inactive / y_pred_active': 'purple',
                                     'y_true_active / y_pred_inactive': 'orange',
                                     'y_true_active / y_pred_active': 'green'
                                 }, alpha=0.7, edgecolor='k', s=100)


            plt.title("t-SNE Visualization of Complete Database", fontsize=16)
            plt.xlabel("Dimension 1", fontsize=14)
            plt.ylabel("Dimension 2", fontsize=14)


            legend = plt.legend(title='Class Label', fontsize=12, title_fontsize=14, loc='best', borderpad=1)
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(1.5)


            plt.grid(True, linestyle='--', linewidth=0.5)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)


            tsne_plot_path_all = os.path.join(results_dir, f"tsne_plot_all.png")
            plt.savefig(tsne_plot_path_all, dpi=300, bbox_inches='tight')
            # plt.show()

            plt.clf()
        messagebox.showinfo("The t-SNE visualization results have been saved to a file:", results_dir)


        try:

            selected_feature_names = X.columns[sfm.get_support()]
            feature_selection_file = os.path.join(results_dir, "selected_features.xlsx")
            pd.DataFrame(selected_feature_names, columns=["Selected Features"]).to_excel(feature_selection_file,
                                                                                         index=False)
            user_response = messagebox.askokcancel("save model", "Do you want to save the model file？")
            if user_response:

                model_dir = os.path.join(file_path, "train_model")
                os.makedirs(model_dir, exist_ok=True)
                model_filePath = os.path.join(model_dir, "model.pkl")
                with open(model_filePath, "wb") as f:
                    pickle.dump(best_rf_model, f)
                messagebox.showinfo("message", "Model has been saved")
            else:

                messagebox.showinfo("message", "Operation has been cancelled.")

        except Exception as e:
            messagebox.showerror("error", f"An error occurred during the model saving process: {e}")

    def GBDT_train_model(self, file_path, algorithm, k_folds):
        file_path_train = os.path.join(file_path, "All_Compound_Descriptor_remove0_removeVif_Standard.xlsx")


        df = pd.read_excel(file_path_train)
        X = df.iloc[:, 2:-1]
        y = df["label"]


        y = y.replace({'active': 1, 'inactive': 0})


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        gbdt_model = GradientBoostingClassifier(random_state=42)


        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

        gbdt_model.fit(X_train,y_train)

        sfm = SelectFromModel(estimator=gbdt_model)
        X_train_selected = sfm.fit_transform(X_train, y_train)
        X_test_selected = sfm.transform(X_test)


        gbdt_model.fit(X_train_selected, y_train)


        y_pred = gbdt_model.predict(X_test_selected)
        y_proba = gbdt_model.predict_proba(X_test_selected)[:, 1]


        accuracy = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        auc_roc = roc_auc_score(y_test, y_proba)
        auc_pr = average_precision_score(y_test, y_proba)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)


        print("After feature selection：")
        print("Selected Features:", X.columns[sfm.get_support()])
        print("Best Accuracy: ", accuracy)
        print("Best MCC: ", mcc)
        print("Best Recall: ", recall)
        print("Best F1 Score: ", f1)
        print("Best Precision: ", precision)
        print("AUC-ROC: ", auc_roc)
        print("AP-PR: ", auc_pr)
        print("Specificity: ", specificity)


        accuracies = []
        mccs = []
        recalls = []
        f1s = []
        precisions = []

        for train_index, test_index in skf.split(X_train_selected, y_train):
            X_train_cv, X_test_cv = X_train_selected[train_index], X_train_selected[test_index]
            y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]


            gbdt_model.fit(X_train_cv, y_train_cv)


            y_pred_cv = gbdt_model.predict(X_test_cv)
            accuracies.append(accuracy_score(y_test_cv, y_pred_cv))
            mccs.append(matthews_corrcoef(y_test_cv, y_pred_cv))
            recalls.append(recall_score(y_test_cv, y_pred_cv))
            f1s.append(f1_score(y_test_cv, y_pred_cv))
            precisions.append(precision_score(y_test_cv, y_pred_cv))


        print("---------------------------------------------------------")
        print(f"After {k_folds}-fold cross-validation:")
        print(f"Mean accuracy over {k_folds}-fold cross-validation: {np.mean(accuracies):.2f}")
        print(f"Mean MCC over {k_folds}-fold cross-validation: {np.mean(mccs):.2f}")
        print(f"Mean Recall over {k_folds}-fold cross-validation: {np.mean(recalls):.2f}")
        print(f"Mean F1 Score over {k_folds}-fold cross-validation: {np.mean(f1s):.2f}")
        print(f"Mean Precision over {k_folds}-fold cross-validation: {np.mean(precisions):.2f}")
        print("------------------------------------------------")


        param_distributions = {
            'n_estimators': randint(50, 200),
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': randint(1, 10),
            'subsample': uniform(0.5, 0.5),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20),
            'max_features': ['auto', 'sqrt', 'log2']
        }

        random_search = RandomizedSearchCV(estimator=gbdt_model, param_distributions=param_distributions,
                                           n_iter=10, cv=skf, scoring='accuracy', random_state=42)

        random_search.fit(X_train_selected, y_train)

        print("Best parameters found: ", random_search.best_params_)
        print("Best accuracy found: ", random_search.best_score_)
        print("------------------------------------------------")

        best_params = random_search.best_params_

        best_gbdt_model = GradientBoostingClassifier(**best_params, random_state=42)

        cv_accuracies = cross_val_score(best_gbdt_model, X_train_selected, y_train, cv=skf, scoring='accuracy')
        cv_mccs = cross_val_score(best_gbdt_model, X_train_selected, y_train, cv=skf, scoring='matthews_corrcoef')
        cv_recalls = cross_val_score(best_gbdt_model, X_train_selected, y_train, cv=skf, scoring='recall')
        cv_f1s = cross_val_score(best_gbdt_model, X_train_selected, y_train, cv=skf, scoring='f1')
        cv_precisions = cross_val_score(best_gbdt_model, X_train_selected, y_train, cv=skf, scoring='precision')
        cv_auc_rocs = cross_val_score(best_gbdt_model, X_train_selected, y_train, cv=skf, scoring='roc_auc')
        cv_ap_prs = cross_val_score(best_gbdt_model, X_train_selected, y_train, cv=skf, scoring='average_precision')


        from sklearn.metrics import make_scorer

        def specificity_score(y_true, y_pred):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tn / (tn + fp)

        specificity_scorer = make_scorer(specificity_score)

        cv_specificities = cross_val_score(best_gbdt_model, X_train_selected, y_train, cv=skf,
                                           scoring=specificity_scorer)

        print("Best parameters found: ", best_params)
        print("Best Accuracy: ", np.mean(cv_accuracies))
        print("Best MCC: ", np.mean(cv_mccs))
        print("Best Recall: ", np.mean(cv_recalls))
        print("Best F1 Score: ", np.mean(cv_f1s))
        print("Best Precision: ", np.mean(cv_precisions))
        print("Best AUC-ROC: ", np.mean(cv_auc_rocs))
        print("Best AP-PR ", np.mean(cv_ap_prs))
        print("Best Specificity: ", np.mean(cv_specificities))

        best_gbdt_model.fit(X_train_selected, y_train)
        y_pred_proba = best_gbdt_model.predict_proba(X_test_selected)[:, 1]
        auc_roc_best = roc_auc_score(y_test, y_pred_proba)
        auc_pr_best = average_precision_score(y_test, y_pred_proba)

        y_pred_best = best_gbdt_model.predict(X_test_selected)
        tn_best, fp_best, fn_best, tp_best = confusion_matrix(y_test, y_pred_best).ravel()
        specificity_best = tn_best / (tn_best + fp_best)

        results_dir = os.path.join(file_path, "results")
        os.makedirs(results_dir, exist_ok=True)

        best_gbdt_model.fit(X_train_selected, y_train)
        y_pred_best = best_gbdt_model.predict(X_test_selected)
        y_pred_best_proba = best_gbdt_model.predict_proba(X_test_selected)[:, 1]

        best_accuracy = accuracy_score(y_test, y_pred_best)
        best_mcc = matthews_corrcoef(y_test, y_pred_best)
        best_recall = recall_score(y_test, y_pred_best)
        best_f1 = f1_score(y_test, y_pred_best)
        best_precision = precision_score(y_test, y_pred_best)

        precision_pr_best, recall_pr_best, _ = precision_recall_curve(y_test, y_pred_best_proba)
        pr_auc_best = auc(recall_pr_best, precision_pr_best)

        fpr_best, tpr_best, _ = roc_curve(y_test, y_pred_best_proba)
        roc_auc_best = auc(fpr_best, tpr_best)


        cm_best = confusion_matrix(y_test, y_pred_best)

        plt.figure()
        sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('GBDT')
        best_cm_path = os.path.join(results_dir, 'Confusion_Matrix.png')
        plt.savefig(best_cm_path)

        plt.clf()


        plt.figure()
        plt.plot(recall_pr_best, precision_pr_best, label=f'PR curve (area = {pr_auc_best:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('GBDT——PR')
        plt.legend(loc="lower right")
        best_pr_curve_path = os.path.join(results_dir, 'PR_curve.png')
        plt.savefig(best_pr_curve_path)

        plt.clf()


        plt.figure()
        plt.plot(fpr_best, tpr_best, label=f'ROC curve (area = {roc_auc_best:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('GBDT——ROC')
        plt.legend(loc="lower right")
        best_roc_curve_path = os.path.join(results_dir, 'ROC_curve.png')
        plt.savefig(best_roc_curve_path)

        plt.clf()




        results = {
            "Feature selection:": {
                "Accuracy": accuracy,
                "MCC": mcc,
                "Recall": recall,
                "F1 Score": f1,
                "Precision": precision,
                "AUC-ROC": auc_roc,
                "AP-PR": auc_pr,
                "Specificity": specificity

            },
            "Feature selection + cross-validation": {
                "Mean Accuracy": np.mean(accuracies),
                "Mean MCC": np.mean(mccs),
                "Mean Recall": np.mean(recalls),
                "Mean F1 Score": np.mean(f1s),
                "Mean Precision": np.mean(precisions),
            },
            "Feature selection + cross-validation + random search：": {
                "Best parameters found": best_params,
                "Mean Accuracy": np.mean(cv_accuracies),
                "Mean MCC": np.mean(cv_mccs),
                "Mean Recall": np.mean(cv_recalls),
                "Mean F1 Score": np.mean(cv_f1s),
                "Mean Precision": np.mean(cv_precisions),
                "Mean AUC-ROC": np.mean(cv_auc_rocs),
                "Mean AP-PR": np.mean(cv_ap_prs),
                "Mean Specificity": np.mean(cv_specificities)
            }

        }

        results_file = os.path.join(results_dir, "results.json")
        with open(results_file, "w", encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        messagebox.showinfo("Training results", f"The result has been saved to the file: {results_file}")

        # # SHAP
        best_gbdt_model.fit(X_train_selected, y_train)


        try:
            explainer = shap.Explainer(best_gbdt_model, X_train_selected)
            shap_values = explainer.shap_values(X_test_selected)
        except TypeError as e:
            print(f"An error occurred while using shap.Explainer: {e}")
            print("Try to use KernelExplainer instead of...")
            explainer = shap.KernelExplainer(best_gbdt_model.predict, X_train_selected)
            shap_values = explainer.shap_values(X_test_selected, nsamples=100)


        results_dir = os.path.join(file_path, "results")
        os.makedirs(results_dir, exist_ok=True)

        selected_feature_names = X.columns[sfm.get_support()]


        if isinstance(X_test_selected, np.ndarray):
            X_test_selected = pd.DataFrame(X_test_selected, columns=selected_feature_names)

        # SHAP Summary Plot (Violin Plot)
        shap.summary_plot(shap_values, X_test_selected, feature_names=selected_feature_names, plot_type="violin",
                          show=False)
        plt.title('SHAP Summary Plot (Violin)')
        plt.xlabel('SHAP Value')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'SHAP_Summary_Violin_Plot.png'))
        # plt.show()

        plt.clf()


        shap.summary_plot(shap_values, X_test_selected, feature_names=selected_feature_names, plot_type="bar",
                          show=False)
        plt.title('SHAP Summary Plot (Bar)')
        plt.xlabel('Mean Absolute SHAP Value')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'SHAP_Summary_Bar_Plot.png'))
        # plt.show()

        plt.clf()


        shap_values_df = pd.DataFrame(shap_values, columns=selected_feature_names)
        plt.figure(figsize=(12, 8))
        sns.heatmap(shap_values_df, cmap='coolwarm', center=0)
        plt.title('SHAP Values Heatmap')
        plt.xlabel('Features')
        plt.ylabel('Samples')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'SHAP_Heatmap.png'))
        # plt.show()

        plt.clf()

        messagebox.showinfo("The SHAP results have been saved to a file: ", results_dir)

        # t-SNE


        data = pd.read_excel(file_path_train, None)


        tsne_perplexity = min(30, X_train_selected.shape[0] - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity)
        X_tsne = tsne.fit_transform(X_train_selected)


        X_tsne_data = pd.DataFrame({
            'Dim1': X_tsne[:, 0],
            'Dim2': X_tsne[:, 1],
            'label': y_train
        })

        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=X_tsne_data, x='Dim1', y='Dim2', hue='label', palette='viridis', alpha=0.7, edgecolor='k',
                        s=100)


        plt.title("t-SNE Visualization of Selected Features", fontsize=16)
        plt.xlabel("Dimension 1", fontsize=14)
        plt.ylabel("Dimension 2", fontsize=14)

        legend = plt.legend(title='Class Label', fontsize=12, title_fontsize=14, loc='best', borderpad=1)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1.5)


        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        tsne_plot_path = os.path.join(results_dir, "tsne_plot_selected.png")
        plt.savefig(tsne_plot_path, dpi=300, bbox_inches='tight')

        plt.clf()

        # plt.show()


        data = pd.read_excel(file_path_train, None)


        for sh_name, sh_data in data.items():

            X = sh_data.drop(['Compound name', 'CAS-number', 'label'], axis=1)
            y_true = sh_data['label']


            y_true_str = y_true.map({0: 'y_true_inactive', 1: 'y_true_active'})

            cat_cols = X.columns.tolist()
            for col in cat_cols:
                X[col] = X[col].astype("category")
                X[col] = X[col].cat.codes


            X_std = StandardScaler().fit_transform(X)


            tsne_perplexity = min(30, X_std.shape[0] - 1)
            tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=42)
            X_tsne = tsne.fit_transform(X_std)


            gbdt_model = GradientBoostingClassifier(random_state=42)
            gbdt_model.fit(X_std, y_true)
            y_pred = gbdt_model.predict(X_std)


            y_pred_str = pd.Series(y_pred).map({0: 'y_pred_inactive', 1: 'y_pred_active'})


            X_tsne_data = pd.DataFrame({
                'Dim1': X_tsne[:, 0],
                'Dim2': X_tsne[:, 1],
                'true_label': y_true_str,
                'pred_label': y_pred_str,
                'combined_label': y_true_str + ' / ' + y_pred_str
            })


            plt.figure(figsize=(12, 10))
            ax = sns.scatterplot(data=X_tsne_data, hue='combined_label', x='Dim1', y='Dim2',
                                 palette={
                                     'y_true_inactive / y_pred_inactive': 'blue',
                                     'y_true_inactive / y_pred_active': 'purple',
                                     'y_true_active / y_pred_inactive': 'orange',
                                     'y_true_active / y_pred_active': 'green'
                                 }, alpha=0.7, edgecolor='k', s=100)


            plt.title("t-SNE Visualization of Complete Database", fontsize=16)
            plt.xlabel("Dimension 1", fontsize=14)
            plt.ylabel("Dimension 2", fontsize=14)


            legend = plt.legend(title='Class Label', fontsize=12, title_fontsize=14, loc='best', borderpad=1)
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(1.5)

            plt.grid(True, linestyle='--', linewidth=0.5)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)


            tsne_plot_path_all = os.path.join(results_dir, f"tsne_plot_all.png")
            plt.savefig(tsne_plot_path_all, dpi=300, bbox_inches='tight')
            # plt.show()

            plt.clf()
        messagebox.showinfo("The t-SNE visualization results have been saved to a file: ", results_dir)


        try:

            selected_feature_names = X.columns[sfm.get_support()]
            feature_selection_file = os.path.join(results_dir, "selected_features.xlsx")
            pd.DataFrame(selected_feature_names, columns=["Selected Features"]).to_excel(feature_selection_file,
                                                                                         index=False)
            user_response = messagebox.askokcancel("save model", "Do you want to save the model file?")
            if user_response:

                model_dir = os.path.join(file_path, "train_model")
                os.makedirs(model_dir, exist_ok=True)
                model_filePath = os.path.join(model_dir, "model.pkl")
                with open(model_filePath, "wb") as f:
                    pickle.dump(best_gbdt_model, f)
                messagebox.showinfo("message", "Model has been saved")
            else:

                messagebox.showinfo("message", "Operation has been cancelled.")

        except Exception as e:
            messagebox.showerror("error", f"An error occurred during the model saving process: {e}")



    def SGD_train_model(self, file_path, algorithm, k_folds):
        file_path_train = os.path.join(file_path, "All_Compound_Descriptor_remove0_removeVif_Standard.xlsx")


        df = pd.read_excel(file_path_train)
        X = df.iloc[:, 2:-1]
        y = df["label"]


        y = y.replace({'active': 1, 'inactive': 0})


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        sgd_model = SGDClassifier( random_state=42)


        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

        sgd_model.fit(X_train,y_train)

        sfm = SelectFromModel(estimator=sgd_model)
        X_train_selected = sfm.fit_transform(X_train, y_train)
        X_test_selected = sfm.transform(X_test)


        sgd_model.fit(X_train_selected, y_train)


        y_pred = sgd_model.predict(X_test_selected)
        # y_proba = sgd_model.predict_proba(X_test_selected)[:, 1]


        accuracy = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        # auc_roc = roc_auc_score(y_test, y_proba)
        # auc_pr = average_precision_score(y_test, y_proba)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)


        print("After feature selection：")
        print("Selected Features:", X.columns[sfm.get_support()])
        print("Best Accuracy: ", accuracy)
        print("Best MCC: ", mcc)
        print("Best Recall: ", recall)
        print("Best F1 Score: ", f1)
        print("Best Precision: ", precision)
        # print("AUC-ROC: ", auc_roc)
        # print("AP-PR: ", auc_pr)
        print("Specificity: ", specificity)


        accuracies = []
        mccs = []
        recalls = []
        f1s = []
        precisions = []


        for train_index, test_index in skf.split(X_train_selected, y_train):
            X_train_cv, X_test_cv = X_train_selected[train_index], X_train_selected[test_index]
            y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]


            sgd_model.fit(X_train_cv, y_train_cv)

            y_pred_cv = sgd_model.predict(X_test_cv)
            accuracies.append(accuracy_score(y_test_cv, y_pred_cv))
            mccs.append(matthews_corrcoef(y_test_cv, y_pred_cv))
            recalls.append(recall_score(y_test_cv, y_pred_cv))
            f1s.append(f1_score(y_test_cv, y_pred_cv))
            precisions.append(precision_score(y_test_cv, y_pred_cv))



        print("---------------------------------------------------------")
        print(f"After {k_folds}-fold cross-validation:")
        print(f"Mean accuracy over {k_folds}-fold cross-validation: {np.mean(accuracies):.2f}")
        print(f"Mean MCC over {k_folds}-fold cross-validation: {np.mean(mccs):.2f}")
        print(f"Mean Recall over {k_folds}-fold cross-validation: {np.mean(recalls):.2f}")
        print(f"Mean F1 Score over {k_folds}-fold cross-validation: {np.mean(f1s):.2f}")
        print(f"Mean Precision over {k_folds}-fold cross-validation: {np.mean(precisions):.2f}")
        print("------------------------------------------------")

        param_distributions = {
            'alpha': loguniform(1e-4, 1e-1),
            'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
            'penalty': ['l2', 'l1', 'elasticnet']
        }


        random_search = RandomizedSearchCV(estimator=sgd_model, param_distributions=param_distributions,
                                           n_iter=10, cv=skf, scoring='accuracy', random_state=42)


        random_search.fit(X_train_selected, y_train)


        print("Best parameters found: ", random_search.best_params_)
        print("Best accuracy found: ", random_search.best_score_)
        print("------------------------------------------------")


        best_params = random_search.best_params_

        if 'loss' not in best_params:
            best_params['loss'] = 'log'


        best_sgd_model = SGDClassifier(**best_params, random_state=42)


        cv_accuracies = cross_val_score(best_sgd_model, X_train_selected, y_train, cv=skf, scoring='accuracy')
        cv_mccs = cross_val_score(best_sgd_model, X_train_selected, y_train, cv=skf, scoring='matthews_corrcoef')
        cv_recalls = cross_val_score(best_sgd_model, X_train_selected, y_train, cv=skf, scoring='recall')
        cv_f1s = cross_val_score(best_sgd_model, X_train_selected, y_train, cv=skf, scoring='f1')
        cv_precisions = cross_val_score(best_sgd_model, X_train_selected, y_train, cv=skf, scoring='precision')
        cv_auc_rocs = cross_val_score(best_sgd_model, X_train_selected, y_train, cv=skf, scoring='roc_auc')
        cv_ap_prs = cross_val_score(best_sgd_model, X_train_selected, y_train, cv=skf, scoring='average_precision')


        from sklearn.metrics import make_scorer

        def specificity_score(y_true, y_pred):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tn / (tn + fp)

        specificity_scorer = make_scorer(specificity_score)

        cv_specificities = cross_val_score(best_sgd_model, X_train_selected, y_train, cv=skf,
                                           scoring=specificity_scorer)

        print("Best parameters found: ", best_params)
        print("Best Accuracy: ", np.mean(cv_accuracies))
        print("Best MCC: ", np.mean(cv_mccs))
        print("Best Recall: ", np.mean(cv_recalls))
        print("Best F1 Score: ", np.mean(cv_f1s))
        print("Best Precision: ", np.mean(cv_precisions))
        print("Best AUC-ROC: ", np.mean(cv_auc_rocs))
        print("Best AP-PR ", np.mean(cv_ap_prs))
        print("Best Specificity: ", np.mean(cv_specificities))

        best_params['loss'] = 'log_loss'
        best_sgd_model = SGDClassifier(**best_params, random_state=42)

        best_sgd_model.fit(X_train_selected, y_train)
        y_pred_proba = best_sgd_model.predict_proba(X_test_selected)[:, 1]
        auc_roc_best = roc_auc_score(y_test, y_pred_proba)
        auc_pr_best = average_precision_score(y_test, y_pred_proba)


        y_pred_best = best_sgd_model.predict(X_test_selected)
        tn_best, fp_best, fn_best, tp_best = confusion_matrix(y_test, y_pred_best).ravel()


        print(f'TN: {tn_best}, FP: {fp_best}, FN: {fn_best}, TP: {tp_best}')


        specificity_best = tn_best / (tn_best + fp_best)
        print(f'Specificity: {specificity_best}')

        results_dir = os.path.join(file_path, "results")
        os.makedirs(results_dir, exist_ok=True)


        best_sgd_model.fit(X_train_selected, y_train)


        y_pred_best = best_sgd_model.predict(X_test_selected)
        try:
            y_pred_best_proba = best_sgd_model.predict_proba(X_test_selected)[:, 1]
        except AttributeError as e:
            print(
                "Error: This 'SGDClassifier' has no attribute 'predict_proba'. Ensure the loss function supports probability estimation.")
            y_pred_best_proba = None


        best_accuracy = accuracy_score(y_test, y_pred_best)
        best_mcc = matthews_corrcoef(y_test, y_pred_best)
        best_recall = recall_score(y_test, y_pred_best)
        best_f1 = f1_score(y_test, y_pred_best)
        best_precision = precision_score(y_test, y_pred_best)


        if y_pred_best_proba is not None:
            precision_pr_best, recall_pr_best, _ = precision_recall_curve(y_test, y_pred_best_proba)
            pr_auc_best = auc(recall_pr_best, precision_pr_best)
        else:
            pr_auc_best=None


        if y_pred_best_proba is not None:
            fpr_best, tpr_best, _ = roc_curve(y_test, y_pred_best_proba)
            roc_auc_best = auc(fpr_best, tpr_best)
        else:
            roc_auc_best=None




        cm_best = confusion_matrix(y_test, y_pred_best)

        plt.figure()
        sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('SGD')
        best_cm_path = os.path.join(results_dir, 'Confusion_Matrix.png')
        plt.savefig(best_cm_path)

        plt.clf()

        if pr_auc_best is not None:

            plt.figure()
            plt.plot(recall_pr_best, precision_pr_best, label=f'PR curve (area = {pr_auc_best:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('SGD——PR')
            plt.legend(loc="lower right")
            best_pr_curve_path = os.path.join(results_dir, 'PR_curve.png')
            plt.savefig(best_pr_curve_path)

            plt.clf()


        if roc_auc_best is not None:
            plt.figure()
            plt.plot(fpr_best, tpr_best, label=f'ROC curve (area = {roc_auc_best:.2f})')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('SGD——ROC')
            plt.legend(loc="lower right")
            best_roc_curve_path = os.path.join(results_dir, 'ROC_curve.png')
            plt.savefig(best_roc_curve_path)

            plt.clf()

        results = {
            "Feature selection：": {
                "Accuracy": accuracy,
                "MCC": mcc,
                "Recall": recall,
                "F1 Score": f1,
                "Precision": precision,
                # "AUC-ROC": auc_roc,
                # "AP-PR": auc_pr,
                "Specificity": specificity

            },
            "Feature selection + cross-validation：": {
                "Mean Accuracy": np.mean(accuracies),
                "Mean MCC": np.mean(mccs),
                "Mean Recall": np.mean(recalls),
                "Mean F1 Score": np.mean(f1s),
                "Mean Precision": np.mean(precisions),
            },
            "Feature selection + cross-validation + random search：": {
                "Best parameters found": best_params,
                "Mean Accuracy": np.mean(cv_accuracies),
                "Mean MCC": np.mean(cv_mccs),
                "Mean Recall": np.mean(cv_recalls),
                "Mean F1 Score": np.mean(cv_f1s),
                "Mean Precision": np.mean(cv_precisions),
                "Mean AUC-ROC": np.mean(cv_auc_rocs),
                "Mean AP-PR": np.mean(cv_ap_prs),
                "Mean Specificity": np.mean(cv_specificities)
            }

        }

        results_file = os.path.join(results_dir, "results.json")
        with open(results_file, "w", encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        messagebox.showinfo("Training results", f"The result has been saved to the file.: {results_file}")

        # SHAP
        best_sgd_model.fit(X_train_selected, y_train)

        try:
            explainer = shap.Explainer(best_sgd_model, X_train_selected)
            shap_values = explainer.shap_values(X_test_selected)
        except TypeError as e:
            print(f"An error occurred while using shap.Explainer: {e}")
            print("Try to use KernelExplainer instead...")
            explainer = shap.KernelExplainer(best_sgd_model.predict, X_train_selected)
            shap_values = explainer.shap_values(X_test_selected, nsamples=100)


        results_dir = os.path.join(file_path, "results")
        os.makedirs(results_dir, exist_ok=True)

        selected_feature_names = X.columns[sfm.get_support()]


        if isinstance(X_test_selected, np.ndarray):
            X_test_selected = pd.DataFrame(X_test_selected, columns=selected_feature_names)

        # SHAP Summary Plot (Violin Plot)
        shap.summary_plot(shap_values, X_test_selected, feature_names=selected_feature_names, plot_type="violin",
                          show=False)
        plt.title('SHAP Summary Plot (Violin)')
        plt.xlabel('SHAP Value')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'SHAP_Summary_Violin_Plot.png'))
        # plt.show()

        plt.clf()

        # SHAP Bar Plot
        shap.summary_plot(shap_values, X_test_selected, feature_names=selected_feature_names, plot_type="bar",
                          show=False)
        plt.title('SHAP Summary Plot (Bar)')
        plt.xlabel('Mean Absolute SHAP Value')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'SHAP_Summary_Bar_Plot.png'))
        # plt.show()

        plt.clf()


        shap_values_df = pd.DataFrame(shap_values, columns=selected_feature_names)
        plt.figure(figsize=(12, 8))
        sns.heatmap(shap_values_df, cmap='coolwarm', center=0)
        plt.title('SHAP Values Heatmap')
        plt.xlabel('Features')
        plt.ylabel('Samples')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'SHAP_Heatmap.png'))
        # plt.show()


        plt.clf()

        messagebox.showinfo("The SHAP results have been saved to a file: ", results_dir)

        # t-SNE
        data = pd.read_excel(file_path_train, None)


        tsne_perplexity = min(30, X_train_selected.shape[0] - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity)
        X_tsne = tsne.fit_transform(X_train_selected)


        X_tsne_data = pd.DataFrame({
            'Dim1': X_tsne[:, 0],
            'Dim2': X_tsne[:, 1],
            'label': y_train
        })


        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=X_tsne_data, x='Dim1', y='Dim2', hue='label', palette='viridis', alpha=0.7, edgecolor='k',
                        s=100)


        plt.title("t-SNE Visualization of Selected Features", fontsize=16)
        plt.xlabel("Dimension 1", fontsize=14)
        plt.ylabel("Dimension 2", fontsize=14)


        legend = plt.legend(title='Class Label', fontsize=12, title_fontsize=14, loc='best', borderpad=1)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1.5)


        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)


        tsne_plot_path = os.path.join(results_dir, "tsne_plot_selected.png")
        plt.savefig(tsne_plot_path, dpi=300, bbox_inches='tight')

        plt.clf()

        # plt.show()



        data = pd.read_excel(file_path_train, None)


        for sh_name, sh_data in data.items():

            X = sh_data.drop(['Compound name', 'CAS-number', 'label'], axis=1)
            y_true = sh_data['label']


            y_true_str = y_true.map({0: 'y_true_inactive', 1: 'y_true_active'})


            cat_cols = X.columns.tolist()
            for col in cat_cols:
                X[col] = X[col].astype("category")
                X[col] = X[col].cat.codes


            X_std = StandardScaler().fit_transform(X)


            tsne_perplexity = min(30, X_std.shape[0] - 1)
            tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=42)
            X_tsne = tsne.fit_transform(X_std)


            sgd_model = SGDClassifier(random_state=42)
            sgd_model.fit(X_std, y_true)
            y_pred = sgd_model.predict(X_std)

            y_pred_str = pd.Series(y_pred).map({0: 'y_pred_inactive', 1: 'y_pred_active'})


            X_tsne_data = pd.DataFrame({
                'Dim1': X_tsne[:, 0],
                'Dim2': X_tsne[:, 1],
                'true_label': y_true_str,
                'pred_label': y_pred_str,
                'combined_label': y_true_str + ' / ' + y_pred_str
            })


            plt.figure(figsize=(12, 10))
            ax = sns.scatterplot(data=X_tsne_data, hue='combined_label', x='Dim1', y='Dim2',
                                 palette={
                                     'y_true_inactive / y_pred_inactive': 'blue',
                                     'y_true_inactive / y_pred_active': 'purple',
                                     'y_true_active / y_pred_inactive': 'orange',
                                     'y_true_active / y_pred_active': 'green'
                                 }, alpha=0.7, edgecolor='k', s=100)


            plt.title("t-SNE Visualization of Complete Database", fontsize=16)
            plt.xlabel("Dimension 1", fontsize=14)
            plt.ylabel("Dimension 2", fontsize=14)


            legend = plt.legend(title='Class Label', fontsize=12, title_fontsize=14, loc='best', borderpad=1)
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(1.5)


            plt.grid(True, linestyle='--', linewidth=0.5)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)


            tsne_plot_path_all = os.path.join(results_dir, f"tsne_plot_all.png")
            plt.savefig(tsne_plot_path_all, dpi=300, bbox_inches='tight')
            # plt.show()

            plt.clf()
        messagebox.showinfo("The t-SNE visualization results have been saved to a file: ", results_dir)


        try:

            selected_feature_names = X.columns[sfm.get_support()]
            feature_selection_file = os.path.join(results_dir, "selected_features.xlsx")
            pd.DataFrame(selected_feature_names, columns=["Selected Features"]).to_excel(feature_selection_file,
                                                                                         index=False)

            user_response = messagebox.askokcancel("save model", "Do you want to save the model file？")
            if user_response:

                model_dir = os.path.join(file_path, "train_model")
                os.makedirs(model_dir, exist_ok=True)
                model_filePath = os.path.join(model_dir, "model.pkl")
                with open(model_filePath, "wb") as f:
                    pickle.dump(best_sgd_model, f)
                messagebox.showinfo("message", "Model has been saved")
            else:

                messagebox.showinfo("message", "Operation has been cancelled.")

        except Exception as e:
            messagebox.showerror("error", f"An error occurred during the model saving process: {e}")


    def GNB_train_model(self, file_path, algorithm, k_folds):
        file_path_train = os.path.join(file_path, "All_Compound_Descriptor_remove0_removeVif_Standard.xlsx")


        df = pd.read_excel(file_path_train)
        X = df.iloc[:, 2:-1]
        y = df["label"]


        y = y.replace({'active': 1, 'inactive': 0})


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)


        selector = VarianceThreshold()
        X_train_selected = selector.fit_transform(X_train_scaled)
        X_test_selected = selector.transform(X_test_scaled)


        gnb_model = GaussianNB()


        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)


        gnb_model.fit(X_train_selected, y_train)


        y_pred = gnb_model.predict(X_test_selected)
        y_proba = gnb_model.predict_proba(X_test_selected)[:, 1]


        accuracy = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        auc_roc = roc_auc_score(y_test, y_proba)
        auc_pr = average_precision_score(y_test, y_proba)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)


        print("After feature selection: ")
        print("Best Accuracy: ", accuracy)
        print("Best MCC: ", mcc)
        print("Best Recall: ", recall)
        print("Best F1 Score: ", f1)
        print("Best Precision: ", precision)
        print("AUC-ROC: ", auc_roc)
        print("AP-PR: ", auc_pr)
        print("Specificity: ", specificity)


        accuracies = []
        mccs = []
        recalls = []
        f1s = []
        precisions = []


        for train_index, test_index in skf.split(X_train_selected, y_train):
            X_train_cv, X_test_cv = X_train_selected[train_index], X_train_selected[test_index]
            y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]


            gnb_model.fit(X_train_cv, y_train_cv)

            y_pred_cv = gnb_model.predict(X_test_cv)
            accuracies.append(accuracy_score(y_test_cv, y_pred_cv))
            mccs.append(matthews_corrcoef(y_test_cv, y_pred_cv))
            recalls.append(recall_score(y_test_cv, y_pred_cv))
            f1s.append(f1_score(y_test_cv, y_pred_cv))
            precisions.append(precision_score(y_test_cv, y_pred_cv))



        print("---------------------------------------------------------")
        print(f"After {k_folds}-fold cross-validation:")
        print(f"Mean accuracy over {k_folds}-fold cross-validation: {np.mean(accuracies):.2f}")
        print(f"Mean MCC over {k_folds}-fold cross-validation: {np.mean(mccs):.2f}")
        print(f"Mean Recall over {k_folds}-fold cross-validation: {np.mean(recalls):.2f}")
        print(f"Mean F1 Score over {k_folds}-fold cross-validation: {np.mean(f1s):.2f}")
        print(f"Mean Precision over {k_folds}-fold cross-validation: {np.mean(precisions):.2f}")

        print("------------------------------------------------")


        param_distributions = {
            'var_smoothing': uniform(1e-9, 1e-6)
        }


        random_search = RandomizedSearchCV(estimator=gnb_model, param_distributions=param_distributions,
                                           n_iter=10, cv=skf, scoring='accuracy', random_state=42)


        random_search.fit(X_train_selected, y_train)


        print("Best parameters found: ", random_search.best_params_)
        print("Best accuracy found: ", random_search.best_score_)
        print("------------------------------------------------")


        best_params = random_search.best_params_


        best_gnb_model = GaussianNB(**best_params)


        cv_accuracies = cross_val_score(best_gnb_model, X_train_selected, y_train, cv=skf, scoring='accuracy')
        cv_mccs = cross_val_score(best_gnb_model, X_train_selected, y_train, cv=skf, scoring='matthews_corrcoef')
        cv_recalls = cross_val_score(best_gnb_model, X_train_selected, y_train, cv=skf, scoring='recall')
        cv_f1s = cross_val_score(best_gnb_model, X_train_selected, y_train, cv=skf, scoring='f1')
        cv_precisions = cross_val_score(best_gnb_model, X_train_selected, y_train, cv=skf, scoring='precision')
        cv_auc_rocs = cross_val_score(best_gnb_model, X_train_selected, y_train, cv=skf, scoring='roc_auc')
        cv_ap_prs = cross_val_score(best_gnb_model, X_train_selected, y_train, cv=skf, scoring='average_precision')


        from sklearn.metrics import make_scorer

        def specificity_score(y_true, y_pred):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tn / (tn + fp)

        specificity_scorer = make_scorer(specificity_score)

        cv_specificities = cross_val_score(best_gnb_model, X_train_selected, y_train, cv=skf,
                                           scoring=specificity_scorer)

        print("Best parameters found: ", best_params)
        print("Best Accuracy: ", np.mean(cv_accuracies))
        print("Best MCC: ", np.mean(cv_mccs))
        print("Best Recall: ", np.mean(cv_recalls))
        print("Best F1 Score: ", np.mean(cv_f1s))
        print("Best Precision: ", np.mean(cv_precisions))
        print("Best AUC-ROC: ", np.mean(cv_auc_rocs))
        print("Best AP-PR ", np.mean(cv_ap_prs))
        print("Best Specificity: ", np.mean(cv_specificities))

        best_gnb_model.fit(X_train_selected, y_train)
        y_pred_proba = best_gnb_model.predict_proba(X_test_selected)[:, 1]
        auc_roc_best = roc_auc_score(y_test, y_pred_proba)
        auc_pr_best = average_precision_score(y_test, y_pred_proba)


        y_pred_best = best_gnb_model.predict(X_test_selected)
        tn_best, fp_best, fn_best, tp_best = confusion_matrix(y_test, y_pred_best).ravel()
        specificity_best = tn_best / (tn_best + fp_best)


        results_dir = os.path.join(file_path, "results")
        os.makedirs(results_dir, exist_ok=True)


        best_gnb_model.fit(X_train_selected, y_train)
        y_pred_best = best_gnb_model.predict(X_test_selected)
        y_pred_best_proba = best_gnb_model.predict_proba(X_test_selected)[:, 1]


        best_accuracy = accuracy_score(y_test, y_pred_best)
        best_mcc = matthews_corrcoef(y_test, y_pred_best)
        best_recall = recall_score(y_test, y_pred_best)
        best_f1 = f1_score(y_test, y_pred_best)
        best_precision = precision_score(y_test, y_pred_best)



        precision_pr_best, recall_pr_best, _ = precision_recall_curve(y_test, y_pred_best_proba)
        pr_auc_best = auc(recall_pr_best, precision_pr_best)


        fpr_best, tpr_best, _ = roc_curve(y_test, y_pred_best_proba)
        roc_auc_best = auc(fpr_best, tpr_best)


        cm_best = confusion_matrix(y_test, y_pred_best)

        plt.figure()
        sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('GNB')
        best_cm_path = os.path.join(results_dir, 'Confusion_Matrix.png')
        plt.savefig(best_cm_path)

        plt.clf()

        plt.figure()
        plt.plot(recall_pr_best, precision_pr_best, label=f'PR curve (area = {pr_auc_best:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('GNB——PR')
        plt.legend(loc="lower right")
        best_pr_curve_path = os.path.join(results_dir, 'PR_curve.png')
        plt.savefig(best_pr_curve_path)

        plt.clf()

        plt.figure()
        plt.plot(fpr_best, tpr_best, label=f'ROC curve (area = {roc_auc_best:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('GNB——ROC')
        plt.legend(loc="lower right")
        best_roc_curve_path = os.path.join(results_dir, 'ROC_curve.png')
        plt.savefig(best_roc_curve_path)

        plt.clf()



        results = {
            "Feature selection：": {
                "Accuracy": accuracy,
                "MCC": mcc,
                "Recall": recall,
                "F1 Score": f1,
                "Precision": precision,
                "AUC-ROC": auc_roc,
                "AP-PR": auc_pr,
                "Specificity": specificity

            },
            "Feature selection + cross-validation：": {
                "Mean Accuracy": np.mean(accuracies),
                "Mean MCC": np.mean(mccs),
                "Mean Recall": np.mean(recalls),
                "Mean F1 Score": np.mean(f1s),
                "Mean Precision": np.mean(precisions),
            },
            "Feature selection + cross-validation + random search：": {
                "Best parameters found": best_params,
                "Mean Accuracy": np.mean(cv_accuracies),
                "Mean MCC": np.mean(cv_mccs),
                "Mean Recall": np.mean(cv_recalls),
                "Mean F1 Score": np.mean(cv_f1s),
                "Mean Precision": np.mean(cv_precisions),
                "Mean AUC-ROC": np.mean(cv_auc_rocs),
                "Mean AP-PR": np.mean(cv_ap_prs),
                "Mean Specificity": np.mean(cv_specificities)
            }

        }

        results_file = os.path.join(results_dir, "results.json")
        with open(results_file, "w", encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        messagebox.showinfo("Training results", f"The result has been saved to the file.: {results_file}")

        # lime
        selected_feature_names = X.columns[selector.get_support()].tolist()

        explainer = LimeTabularExplainer(
            X_train_selected,
            mode="classification",
            feature_names=selected_feature_names,
            discretize_continuous=True
        )


        idx = random.randint(0, X_test_selected.shape[0] - 1)
        sample_data_array = X_test_selected[idx]


        exp = explainer.explain_instance(
            sample_data_array,
            gnb_model.predict_proba,
            num_features=min(50, len(selected_feature_names)),
            top_labels=2,
            num_samples=1000
        )


        exp.save_to_file(os.path.join(results_dir, 'lime_explanation.html'))


        lime_results = {
            "instance_index": str(idx),
            "predicted_class": int(gnb_model.predict(sample_data_array.reshape(1, -1))[0]),
            "lime_explanation": {
                "as_list": exp.as_list(),
                "as_map": exp.as_map(),
                "predict_proba": gnb_model.predict_proba(sample_data_array.reshape(1, -1)).tolist()
            }
        }

        lime_results_file = os.path.join(results_dir, "lime_explanation.txt")
        with open(lime_results_file, "w", encoding="utf-8") as f:
            f.write("Lime result\n\n")
            f.write(f"Example Index: {lime_results['instance_index']}\n")
            f.write(f"Prediction category: {lime_results['predicted_class']}\n\n")

            f.write("Lime result:\n")
            f.write("As List:\n")
            for feature, weight in lime_results['lime_explanation']['as_list']:
                f.write(f"{feature}: {weight}\n")

            f.write("\nAs Map:\n")
            for label, weights in lime_results['lime_explanation']['as_map'].items():
                f.write(f"Label: {label}\n")
                for feature, weight in weights:
                    f.write(f"{feature}: {weight}\n")

            f.write("\nPredicted Probabilities:\n")
            for prob in lime_results['lime_explanation']['predict_proba']:
                f.write(f"{prob}\n")

        messagebox.showinfo("Lime Interpretation",
                            f"The LIME explanation results have been saved in: {results_dir}")

        # t-SNE
        data = pd.read_excel(file_path_train, None)


        tsne_perplexity = min(30, X_train_selected.shape[0] - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity)
        X_tsne = tsne.fit_transform(X_train_selected)


        X_tsne_data = pd.DataFrame({
            'Dim1': X_tsne[:, 0],
            'Dim2': X_tsne[:, 1],
            'label': y_train
        })


        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=X_tsne_data, x='Dim1', y='Dim2', hue='label', palette='viridis', alpha=0.7, edgecolor='k',
                        s=100)

        plt.title("t-SNE Visualization of Selected Features", fontsize=16)
        plt.xlabel("Dimension 1", fontsize=14)
        plt.ylabel("Dimension 2", fontsize=14)

        legend = plt.legend(title='Class Label', fontsize=12, title_fontsize=14, loc='best', borderpad=1)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1.5)


        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)


        tsne_plot_path = os.path.join(results_dir, "tsne_plot_selected.png")
        plt.savefig(tsne_plot_path, dpi=300, bbox_inches='tight')

        plt.clf()

        # plt.show()



        data = pd.read_excel(file_path_train, None)


        for sh_name, sh_data in data.items():

            X = sh_data.drop(['Compound name', 'CAS-number', 'label'], axis=1)
            y_true = sh_data['label']


            y_true_str = y_true.map({0: 'y_true_inactive', 1: 'y_true_active'})


            cat_cols = X.columns.tolist()
            for col in cat_cols:
                X[col] = X[col].astype("category")
                X[col] = X[col].cat.codes


            X_std = StandardScaler().fit_transform(X)


            tsne_perplexity = min(30, X_std.shape[0] - 1)
            tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=42)
            X_tsne = tsne.fit_transform(X_std)


            gnb_model = GaussianNB()
            gnb_model.fit(X_std, y_true)
            y_pred = gnb_model.predict(X_std)


            y_pred_str = pd.Series(y_pred).map({0: 'y_pred_inactive', 1: 'y_pred_active'})


            X_tsne_data = pd.DataFrame({
                'Dim1': X_tsne[:, 0],
                'Dim2': X_tsne[:, 1],
                'true_label': y_true_str,
                'pred_label': y_pred_str,
                'combined_label': y_true_str + ' / ' + y_pred_str
            })


            plt.figure(figsize=(12, 10))
            ax = sns.scatterplot(data=X_tsne_data, hue='combined_label', x='Dim1', y='Dim2',
                                 palette={
                                     'y_true_inactive / y_pred_inactive': 'blue',
                                     'y_true_inactive / y_pred_active': 'purple',
                                     'y_true_active / y_pred_inactive': 'orange',
                                     'y_true_active / y_pred_active': 'green'
                                 }, alpha=0.7, edgecolor='k', s=100)


            plt.title("t-SNE Visualization of Complete Database", fontsize=16)
            plt.xlabel("Dimension 1", fontsize=14)
            plt.ylabel("Dimension 2", fontsize=14)


            legend = plt.legend(title='Class Label', fontsize=12, title_fontsize=14, loc='best', borderpad=1)
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(1.5)


            plt.grid(True, linestyle='--', linewidth=0.5)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)


            tsne_plot_path_all = os.path.join(results_dir, f"tsne_plot_all.png")
            plt.savefig(tsne_plot_path_all, dpi=300, bbox_inches='tight')
            # plt.show()

            plt.clf()
        messagebox.showinfo("The t-SNE visualization results have been saved to a file: ", results_dir)


        try:

            selected_feature_names = X.columns[selector.get_support()]
            feature_selection_file = os.path.join(results_dir, "selected_features.xlsx")
            pd.DataFrame(selected_feature_names, columns=["Selected Features"]).to_excel(feature_selection_file,
                                                                                         index=False)
            user_response = messagebox.askokcancel("save model", "Do you want to save the model file？")
            if user_response:

                model_dir = os.path.join(file_path, "train_model")
                os.makedirs(model_dir, exist_ok=True)
                model_filePath = os.path.join(model_dir, "model.pkl")
                with open(model_filePath, "wb") as f:
                    pickle.dump(best_gnb_model, f)
                messagebox.showinfo("message", "Model has been saved")
            else:

                messagebox.showinfo("message", "Operation has been cancelled.")

        except Exception as e:
            messagebox.showerror("error", f"An error occurred during the model saving process: {e}")



class DatabaseViewer:
    def __init__(self, master):
        self.master = master
        master.title("Please select the system database you wish to view: ")
        master.geometry("600x350")

        master.configure(bg="#f0f0f0")

        database_button_width=30

        self.button1 = tk.Button(master, text="Compound_OP_binding.xlsx",
                                 command=lambda: self.view_file(
                                     "navicat_to_excel/Compound_OP_binding.xlsx"),
                                 font=('Helvetica', 12), bg="#2196F3", fg="white",width=database_button_width)
        self.button1.pack(pady=10)



        self.button3 = tk.Button(master, text="Compound_info_1.xlsx",
                                 command=lambda: self.view_file(
                                     "navicat_to_excel/Compound_info_1.xlsx"),
                                 font=('Helvetica', 12), bg="#2196F3", fg="white",width=database_button_width)
        self.button3.pack(pady=10)



        self.button5 = tk.Button(master, text="All_Compound_Descriptor.xlsx",
                                 command=lambda: self.view_file(
                                     "navicat_to_excel/All_Compound_Descriptor.xlsx"),
                                 font=('Helvetica', 12), bg="#2196F3", fg="white",width=database_button_width)
        self.button5.pack(pady=10)

        self.button6 = tk.Button(master, text="Descriptor_vif_results.xlsx",
                                 command=lambda: self.view_file(
                                     "navicat_to_excel/Descriptor_vif_results.xlsx"),
                                 font=('Helvetica', 12), bg="#2196F3", fg="white", width=database_button_width)
        self.button6.pack(pady=10)

    def view_file(self, file_name):
        try:
            df = pd.read_excel(file_name)
            file_window = tk.Toplevel(self.master)
            file_window.title(os.path.basename(file_name))
            file_window.geometry("800x600")

            text = tk.Text(file_window, wrap='none', font=('Courier', 10))
            text.insert(tk.END, df.to_string())
            text.pack(expand=True, fill='both')

            xscrollbar = tk.Scrollbar(file_window, orient=tk.HORIZONTAL, command=text.xview)
            xscrollbar.pack(side=tk.BOTTOM, fill=tk.X)
            yscrollbar = tk.Scrollbar(file_window, orient=tk.VERTICAL, command=text.yview)
            yscrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            text.config(xscrollcommand=xscrollbar.set, yscrollcommand=yscrollbar.set)

        except FileNotFoundError:
            messagebox.showerror("File not found", f"{file_name} not found.")
        except Exception as e:
            messagebox.showerror("error", str(e))


class PredictWindow:
    def __init__(self, master):
        self.master = master
        logger.info("Open the prediction window")
        master.title("Prediction window: ")
        master.geometry("700x600")
        master.configure(bg="#f0f0f0")

        self.center_frame = tk.Frame(master, bg="#f0f0f0")
        self.center_frame.place(relx=0.5, rely=0.5, anchor='center')


        self.model_label2 = tk.Label(self.center_frame, text="Select the model file", font=('Helvetica', 12), bg="#f0f0f0")
        self.model_label2.grid(row=0, column=0, padx=10, pady=10, sticky='e')
        self.model_entry2 = tk.Entry(self.center_frame, font=('Helvetica', 12), width=30)
        self.model_entry2.grid(row=0, column=1, padx=10, pady=10, sticky='w')
        self.model_button = tk.Button(self.center_frame, text="Browse", command=self.browse_model_file)
        self.model_button.grid(row=0, column=2, pady=10, sticky='w')

        self.model_label1 = tk.Label(self.center_frame, text="Compound name/CAS number/SMILES", font=('Helvetica', 12), bg="#f0f0f0")
        self.model_label1.grid(row=1, column=0, padx=10, pady=10, sticky='e')
        self.model_entry = tk.Entry(self.center_frame, font=('Helvetica', 12), width=30)
        self.model_entry.grid(row=1, column=1, padx=10, pady=10, sticky='w')

        self.predict_button = tk.Button(self.center_frame, text="Single prediction", font=('Helvetica', 14, 'bold'), bg="#4CAF50",
                                        fg="white", command=self.predict)
        self.predict_button.grid(row=2, column=0, columnspan=2, pady=10)

        self.upload_label = tk.Label(self.center_frame, text="Select the file to be predicted", font=('Helvetica', 12), bg="#f0f0f0")
        self.upload_label.grid(row=3, column=0, padx=10, pady=10, sticky='e')
        self.upload_entry = tk.Entry(self.center_frame, font=('Helvetica', 12), width=30)
        self.upload_entry.grid(row=3, column=1, padx=10, pady=10, sticky='w')
        self.upload_button = tk.Button(self.center_frame, text="Browse", command=self.browse_upload_file)
        self.upload_button.grid(row=3, column=2, pady=10, sticky='w')

        self.batch_predict_button = tk.Button(self.center_frame, text="Batch prediction", font=('Helvetica', 14, 'bold'), bg="#4CAF50",
                                              fg="white", command=self.batch_predict)
        self.batch_predict_button.grid(row=4, column=0, columnspan=2, pady=10)

    def browse_model_file(self):
        model_file = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if model_file:
            self.model_entry2.insert(0, model_file)
        logger.info(f"Model file has been selected: {model_file}")

    def browse_upload_file(self):
        upload_file = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if upload_file:
            self.upload_entry.insert(0, upload_file)
        logger.info(f"File upload selected: {upload_file}")



    def predict(self):
        compound_information = self.model_entry.get().strip()
        model_file_path = self.model_entry2.get()

        if not compound_information or not model_file_path:
            messagebox.showerror("error", "Please provide the compound information and train the model.")
            logger.error("Lack of compound information or model files")
            return

        file_path = os.path.dirname(os.path.dirname(model_file_path))

        data_file_path = os.path.join(file_path, "All_Compound_Descriptor_remove0_removeVif.xlsx")
        selected_feature_path = os.path.join(file_path, "results/selected_features.xlsx")
        scaler_path = os.path.join(file_path, "results/scaler.pkl")

        try:
            with open(model_file_path, "rb") as f:
                model = pickle.load(f)
                logger.info("Model file loading successful")
        except Exception as e:
            messagebox.showerror("error", f"An error occurred while loading the model: {e}")
            logger.error(f"An error occurred while loading the model: {e}")
            return

        try:

            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)


            df_data = pd.read_excel(data_file_path)
            df_features = pd.read_excel(selected_feature_path)
            selected_feature = df_features['Selected Features'].tolist()


            selected_feature = [name.strip().strip("'").strip() for name in selected_feature]


            feature_columns = df_data.columns[2:-1].tolist()

            feature_columns = [name.strip().strip("'").strip() for name in feature_columns]


            from rdkit import Chem
            mol = Chem.MolFromSmiles(compound_information)
            if mol is not None:

                smiles = compound_information

            else:

                logger.info(f"The input is not SMILES. Please try to obtain the SMILES for the compound {compound_information}.")
                smiles = enhanced_smiles_retrieval(compound_information)
                if not smiles:
                    messagebox.showerror("error", "Unable to find the SMILES of the compound.")
                    logger.warning(f"Unable to find the SMILES of the compound: {compound_information}")
                    return


            descriptors = compute_descriptors(smiles, feature_columns)
            if not descriptors:
                messagebox.showerror("error", "The descriptors of the compound cannot be calculated.")
                logger.warning(f"The descriptors of the compound cannot be calculated: {compound_information}")
                return


            X = pd.DataFrame([descriptors])[feature_columns].values


            X_scaled = scaler.transform(X)


            X_selected = X_scaled[:, [feature_columns.index(f) for f in selected_feature]]


            predictions = model.predict(X_selected)
            result = "active" if predictions[0] == 1 else "inactive"
            messagebox.showinfo("Compound", f"{compound_information} Forecasting outcome: {result}")
            logger.info(f"{compound_information} Forecasting outcome: {result}")

        except Exception as e:
            messagebox.showerror("error", f"Errors occurred during the prediction process: {e}")
            logger.error(f"Errors occurred during the prediction process: {e}")

    def batch_predict(self):
        upload_file_path = self.upload_entry.get()
        model_file_path = self.model_entry2.get()

        if not upload_file_path or not model_file_path:
            messagebox.showerror("error", "Please provide the file to be predicted and the training model.")
            logger.error("Missing the file or model file to be predicted")
            return

        file_path = os.path.dirname(os.path.dirname(model_file_path))

        data_file_path = os.path.join(file_path, "All_Compound_Descriptor_remove0_removeVif.xlsx")
        selected_feature_path = os.path.join(file_path, "results/selected_features.xlsx")
        scaler_path = os.path.join(file_path, "results/scaler.pkl")

        try:
            with open(model_file_path, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            messagebox.showerror("error", f"An error occurred while loading the model. {e}")
            logger.error(f"An error occurred while loading the model. {e}")
            return

        try:

            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

            df_upload = pd.read_excel(upload_file_path)

            df_data = pd.read_excel(data_file_path)
            df_features = pd.read_excel(selected_feature_path)
            selected_feature = df_features['Selected Features'].tolist()

            selected_feature = [name.strip().strip("'").strip() for name in selected_feature]


            feature_columns = df_data.columns[2:-1].tolist()

            feature_columns = [name.strip().strip("'").strip() for name in feature_columns]

            results = []


            for compound_name in df_upload['Compound information']:

                from rdkit import Chem
                mol = Chem.MolFromSmiles(compound_name)
                if mol is not None:

                    smiles = compound_name

                else:

                    logger.info(f"Attempt to obtain the SMILES of the compound {compound_name}")
                    smiles = enhanced_smiles_retrieval(compound_name)
                    if not smiles:
                        logger.warning(f"The SMILES for the compound {compound_name} could not be found.")
                        results.append((compound_name, "not found"))
                        continue


                descriptors = compute_descriptors(smiles, feature_columns)
                if not descriptors:
                    logger.warning(f"The descriptors of the compound {compound_name} cannot be calculated.")
                    results.append((compound_name, "descriptor calculation error"))
                    continue


                X = pd.DataFrame([descriptors])[feature_columns].values


                X_scaled = scaler.transform(X)


                X_selected = X_scaled[:, [feature_columns.index(f) for f in selected_feature]]


                predictions = model.predict(X_selected)
                result = "1" if predictions[0] == 1 else "0"
                results.append((compound_name, result))
                logger.info(f"{compound_name}预测结果: {result}")


            df_results = pd.DataFrame(results, columns=["Compound information", "label_predict"])
            save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
            if save_path:
                df_results.to_excel(save_path, index=False)
                messagebox.showinfo("Success", f"The prediction results have been saved to the file：{save_path}")
                logger.info(f"The prediction results have been saved to the file.{save_path}")

        except Exception as e:
            messagebox.showerror("error", f"Error occurred during the prediction process: {e}")
            logger.error(f"Errors occurred during the prediction process: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = WelcomeApp(root)
    root.mainloop()
