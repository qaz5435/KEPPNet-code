"""
GMT file reader for Reactome pathway-gene mappings.
GMT format: each line is <pathway_id>\t<description>\t<gene1>\t<gene2>\t...
"""

import pandas as pd


class GMT:
    def load_data(self, filename: str, pathway_col: int = 0, genes_col: int = 2) -> pd.DataFrame:
        """
        Load a GMT file and return a DataFrame with columns ['gene', 'group'].

        Args:
            filename: path to the .gmt file
            pathway_col: column index of the pathway ID (0-indexed)
            genes_col: column index where genes start (0-indexed)

        Returns:
            DataFrame with columns ['gene', 'group']
        """
        rows = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < genes_col + 1:
                    continue
                pathway_id = parts[pathway_col]
                genes = parts[genes_col:]
                for gene in genes:
                    gene = gene.strip()
                    if gene:
                        rows.append({'gene': gene, 'group': pathway_id})
        return pd.DataFrame(rows, columns=['gene', 'group'])
