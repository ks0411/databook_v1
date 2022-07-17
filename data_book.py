import pandas as pd
import string
import numpy as np
import re
from pathlib import Path

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
        self.df_all = None
        self.df = None
    
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

        df_all['key']=df_all['sheetName']+'!'+df_all['cellAddress']
        df_all = df_all.set_index('key')


        self.df_all = df_all[self.metadata.keys()]
    
    def load_data(self, df: pd.DataFrame):
        '''
            Load a dataframe directly, for test purposes
            This is only checking a df is passed
        '''
        assert(isinstance(df,pd.DataFrame))            
        self.df_all = df
    
    def pre_process_data(self):
        '''
        Run pre-processing of data
        '''
        def calc_feature(r, k1, k2, f):
            this = self._get_that_from_this(r, k1)
            that = self._get_that_from_this(r, k2)
            return f(this, that)
        
        self.df_all['vNormFormula'] = self.df_all.apply(lambda r: v_norm_formula(r['cellFormula']), axis=1)
        self.df_all['Label'] = False
        self.df = self.df_all[self.df_all.cellFormula.notnull()].copy(deep=True)

        self.df['up1_isBlank'] = self.df.apply(lambda r: calc_feature(r, 0, -1, self._isBlank), axis=1)
        self.df['up1_isFormula'] = self.df.apply(lambda r: calc_feature(r, 0, -1, self._isFormula), axis=1)
        self.df['up1_isSameType'] = self.df.apply(lambda r: calc_feature(r, 0, -1, self._isSameType), axis=1)
        self.df['up1_isWeaklyFormulaConsistent'] = self.df.apply(lambda r: calc_feature(r, 0, -1, self._isWeaklyFormulaConsistent), axis=1)
        self.df['up2_isWeaklyFormulaConsistent'] = self.df.apply(lambda r: calc_feature(r, -1, -2, self._isWeaklyFormulaConsistent), axis=1)

        self.df['dw1_isBlank'] = self.df.apply(lambda r: calc_feature(r, 0, 1, self._isBlank), axis=1)
        self.df['dw1_isFormula'] = self.df.apply(lambda r: calc_feature(r, 0, 1, self._isFormula), axis=1)
        self.df['dw1_isSameType'] = self.df.apply(lambda r: calc_feature(r, 0, 1, self._isSameType), axis=1)
        self.df['dw1_isWeaklyFormulaConsistent'] = self.df.apply(lambda r: calc_feature(r, 0, 1, self._isWeaklyFormulaConsistent), axis=1)
        self.df['dw2_isWeaklyFormulaConsistent'] = self.df.apply(lambda r: calc_feature(r, -1, -2, self._isWeaklyFormulaConsistent), axis=1)
        self.df['nb1_isWeaklyFormulaConsistent'] = self.df.apply(lambda r: calc_feature(r, -1, 1, self._isWeaklyFormulaConsistent), axis=1)
        self.df['dw1_isSum'] = self.df.apply(lambda r: calc_feature(r, 0, 1, self._isSum), axis=1)


    def get_source_data(self):
        '''
        returns source data 
        '''
        return self.df_all
    
    def get_data(self, all_columns=False):
        '''
        return pre-processed data
        set all_columns to true to get all columns
        '''
        cols_to_drop = ['sheetName', 
                        'numRow', 
                        'numCol', 
                        'cellValue', 
                        'cellAddress', 
                        'workbookName',
                        'cellFormula', 
                        'colHeader', 
                        'rowHeader', 
                        'cellType', 
                        'vNormFormula']
        if all_columns:
            return self.df
        else:
            return self.df[[c for c in self.df.columns if c not in cols_to_drop]]
    
    def get_inconsistent_cells(self):
        df=self.get_data(all_columns=False).copy(deep=True)
        features = [c for c in df.columns if c not in ['Label', 'key', 'lineageId', 'SheetName']]
        df = df[features]
        df=df[(~df['dw1_isWeaklyFormulaConsistent'])|(~df['up1_isWeaklyFormulaConsistent'])]
#         df.reset_index(inplace=True)
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
    
    def add_negative_cases(self, file_path:str):
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
        
        self._add_negative_cases(keys)
                    
    def _add_negative_cases(self, keys):
           
        df2 = self.df.filter(items=keys, axis=0).copy(deep=True)
        df2['Label']=True
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
            cell_ref = self._get_v_cell_ref(this['numRow'], this['numCol'], k, this['sheetName'])
            try:
                that = self.df_all.loc[cell_ref]
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
        
