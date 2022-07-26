import re
import string
from pathlib import Path

import numpy as np
import pandas as pd


def v_norm_formula(formula: str):
    '''
    Normalize a formula by changing any number to *
    e.g., A5:D13 -> A*:D*
    '''
    formula = '' if pd.isnull(formula) else formula
    return re.sub('[0-9]+', '*', formula)

def col_names():
    '''
    Returns names of excel columns in a list, from A to BZ
    '''
    names = ['na']
    names.extend(list(string.ascii_uppercase))
    base_names = list(string.ascii_uppercase)
    for out_name in ['A', 'B']:
        new_names = []
        for in_name in base_names:
            new_names.append(f"{out_name}{in_name}")
        names.extend(new_names)
    return names

class DataBook():
    def __init__(self, raw_data_path: str = None, metadata: dict = None):
        '''
        Class to pre-process data
        raw_data_path: path to excel file
        metadata: mapping of expected columns names to real column names. Different names will be ignored.
        '''
        
        defaultValues = {
                            'workbookName': 'Workbook',
                            'sheetName': 'SheetName',
                            'numRow': 'RowNum',
                            'numCol': 'ColNum',
                            'cellAddress': 'CellAddress',
                            'cellValue': 'Label',
                            'cellFormula': 'Formula',
                            'cellType': 'DataType'
                        }        
        
        metadata = metadata or defaultValues
        assert metadata.keys() == defaultValues.keys(), 'Wrong metadata has been passed'
        
        self.metadata = metadata
        self.raw_data_path = None
        self.df_source = None
        self.df = None # this is used for preprocessed data

        self.feature_names = None 
        self.label = 'Label'
        self.sheet_name = 'sheetName'
        self.cell_address = 'cellAddress'
    
    def load_file(self, raw_data_path: str):
        '''
        Load a file if metadata compatible        
        '''
        path = Path(raw_data_path)
        suffix = path.suffix
        
        assert path.is_file(), f"Failed to load {path}"
        assert (suffix in ['.xls', '.xlsx', '.json']), f"{suffix} is not a supported file type"
        
        if suffix == '.json':
            df_all = pd.read_json(raw_data_path)        
        else:
            df_all = pd.read_excel(raw_data_path)
          
        self.raw_data_path = path
        
        for _, col in self.metadata.items():
            assert (col in df_all.columns), f'{col} is not available in data'

        df_all.rename(columns = {v:k for k,v in self.metadata.items()}, inplace = True)

        assert (len(df_all['workbookName'].unique())==1), 'Only 1 workbook is supported so far!'

        df_all['key']=df_all[self.sheet_name]+'!'+df_all[self.cell_address]
        df_all = df_all.set_index('key')


        self.df_source = df_all[self.metadata.keys()]
    
    def load_data(self, df: pd.DataFrame):
        '''
            Load a dataframe directly, for test purposes
            This is only checking a df is passed
        '''
        assert(isinstance(df,pd.DataFrame))            
        self.df_source = df
    
    def pre_process_data(self, for_training=True):
        '''
        Run pre-processing of data
        '''
        def calc_feature(r, k1, k2, f):
            this = self._get_that_from_this(r, k1)
            that = self._get_that_from_this(r, k2)
            return f(this, that)
        
        self.df_source['vNormFormula'] = self.df_source.apply(lambda r: v_norm_formula(r['cellFormula']), axis=1)
        
        if for_training:
            self.df_source[self.label] = False

        self.df = self.df_source[self.df_source.cellFormula.notnull()].copy(deep=True)
        self.feature_names = []

        self.df['up1_isBlank'] = self.df.apply(lambda r: calc_feature(r, 0, -1, self._isBlank), axis=1)
        self.feature_names.append('up1_isBlank')

        self.df['up1_isFormula'] = self.df.apply(lambda r: calc_feature(r, 0, -1, self._isFormula), axis=1)
        self.feature_names.append('up1_isFormula')

        self.df['up1_isSameType'] = self.df.apply(lambda r: calc_feature(r, 0, -1, self._isSameType), axis=1)
        self.feature_names.append('up1_isSameType')

        self.df['up1_isWeaklyFormulaConsistent'] = self.df.apply(lambda r: calc_feature(r, 0, -1, self._isWeaklyFormulaConsistent), axis=1)
        self.feature_names.append('up1_isWeaklyFormulaConsistent')

        self.df['up2_isWeaklyFormulaConsistent'] = self.df.apply(lambda r: calc_feature(r, -1, -2, self._isWeaklyFormulaConsistent), axis=1)
        self.feature_names.append('up2_isWeaklyFormulaConsistent')

        self.df['dw1_isBlank'] = self.df.apply(lambda r: calc_feature(r, 0, 1, self._isBlank), axis=1)
        self.feature_names.append('dw1_isBlank')

        self.df['dw1_isFormula'] = self.df.apply(lambda r: calc_feature(r, 0, 1, self._isFormula), axis=1)
        self.feature_names.append('dw1_isFormula')

        self.df['dw1_isSameType'] = self.df.apply(lambda r: calc_feature(r, 0, 1, self._isSameType), axis=1)
        self.feature_names.append('dw1_isSameType')

        self.df['dw1_isWeaklyFormulaConsistent'] = self.df.apply(lambda r: calc_feature(r, 0, 1, self._isWeaklyFormulaConsistent), axis=1)
        self.feature_names.append('dw1_isWeaklyFormulaConsistent')

        self.df['dw2_isWeaklyFormulaConsistent'] = self.df.apply(lambda r: calc_feature(r, -1, -2, self._isWeaklyFormulaConsistent), axis=1)
        self.feature_names.append('dw2_isWeaklyFormulaConsistent')

        self.df['nb1_isWeaklyFormulaConsistent'] = self.df.apply(lambda r: calc_feature(r, -1, 1, self._isWeaklyFormulaConsistent), axis=1)
        self.feature_names.append('nb1_isWeaklyFormulaConsistent')

        self.df['dw1_isSum'] = self.df.apply(lambda r: calc_feature(r, 0, 1, self._isSum), axis=1)
        self.feature_names.append('dw1_isSum')
        

    def get_data(self, all_columns=False):
        '''
        return pre-processed data
        set all_columns to true to get all columns
        '''
        if self.df is None:
            df = self.df_source
        else:
            df = self.df 

        if all_columns:
            return df
        else:
            cols_to_drop = [ 
                        'numRow', 
                        'numCol', 
                        'cellValue', 
                        'workbookName',
                        'cellFormula', 
                        'colHeader', 
                        'rowHeader', 
                        'cellType', 
                        'vNormFormula']
            return df[[c for c in self.df.columns.to_list() if c not in cols_to_drop]]

    def get_inconsistent_cells(self, sheet_filter=None, cell_filter=None):
        '''
            Returns cells with an inconsistent formula, to be used in inference.
            sheet_filter and cell_filter filter the df over sheet name or cell address.
        '''
        columns = [self.sheet_name]
        columns.append(self.cell_address)
        columns.extend(self.feature_names)
        if self.label in self.df.columns.to_list():
            columns.append(self.label)
        df=self.df[columns].copy(deep=True)

        df=df[(~df['dw1_isWeaklyFormulaConsistent'])|(~df['up1_isWeaklyFormulaConsistent'])]

        if sheet_filter:
            df = df[df[self.sheet_name]==sheet_filter]
        if cell_filter:
            df = df[df[self.cell_address]==cell_filter]

        # df = df.set_index('key')
        df = df[self.feature_names]
        

        return df
    
    def get_raw_data_path(self):
        '''
        returns the path to raw data
        '''
        return self.raw_data_path
    
    def get_column_mappings(self):
        '''
        returns metadata
        '''
        return self.metadata
    
    def add_positive_cases(self, file_path:str):
        '''
        Add records representing corrupted cells from a file
        
        file_path: a TSV file with the following columns, representing ranges to corrupt
            sheet name, comma separated list of column names, rows range
                e.g., sheet1 \t A,B,C \t 3,9 \n
        '''
        with open(file_path) as f:
            lines = f.readlines()
           
        keys=[]
           
        for line in lines:
            line=line[:-1]
            fields = line.split('\t')
            
            sheet_name = fields[0].strip()
            for col in fields[1].split(','):
                row_start_end = fields[2].split(',')
                x=int(row_start_end[0])
                y=int(row_start_end[1])+1
                for row in range(x,y):
                    keys.append(f"{sheet_name}!{col}{row}")
        
        self._add_positive_cases(keys)
                    
    def _add_positive_cases(self, keys):
           
        df2 = self.df.filter(items=keys, axis=0).copy(deep=True)
        df2[self.label]=True
        df2['up1_isWeaklyFormulaConsistent']=False
        df2['dw1_isWeaklyFormulaConsistent']=False
        self.df=pd.concat([self.df,df2])
    
    
    
    ### internal functions
    
    def _get_v_cell_ref(self, i:int, j:int, k:int, sheet_name:str): 
        '''
        Returns a cell name that is k vertical positions over a given cell (i,j)
        '''
        i = i+k
        if i >= 1:
            col_name = col_names()[j]
            return f"{sheet_name}!{col_name}{i}"
        else:
            return None
    
    def _get_that_from_this(self, this, k):
        '''
        Returns a cell that is k vertical positions over a given cell (i,j)
        '''
        if k==0:
            return this
        else:
            cell_ref = self._get_v_cell_ref(this['numRow'], this['numCol'], k, this[self.sheet_name])
            try:
                that = self.df_source.loc[cell_ref]
            except:
                that = None
            return that
    
    def _isBlank(self, this, that):
        if that is None:
            # if the v-next cell is not tracked in dataset, we consider it as blank
            return True
        else:
            return pd.isna(that['cellValue'])

    def _isFormula(self, this, that):
        if that is None:
            # if the v-next cell is not tracked in dataset, we consider it as missing formula
            return False
        else:
            return not pd.isna(that['cellFormula'])
        
    def _isSameType(self, this, that):
        if that is None or this is None:
            # if the v-next cell is not tracked in dataset, we consider it as a different type
            return False        
        else:
            return that['cellType']==this['cellType']

    def _isWeaklyFormulaConsistent(self, this, that):
        if that is None or this is None:
            # if the v-next cell is not tracked in dataset, we consider it as weakly consistent
            return True
        else:
            return this['vNormFormula']==that['vNormFormula']
    
    def _isSum(self, this, that):
        if that is None:
            return False
        else:
            formula = that['vNormFormula']
            if len(formula)>5:
                return formula[0:4].lower()=='sum('
            else:
                return False
        
