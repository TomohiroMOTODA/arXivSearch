from datetime import datetime
from collections import Counter
import re
import time
import requests
import yaml
import numpy as np

# Check for necessary library installations
try:
    import arxiv
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from wordcloud import WordCloud
    import xml.etree.ElementTree as ET
    from urllib.parse import quote
    from tqdm import tqdm
    from datetime import timedelta
except ImportError as e:
    print(f"Required libraries are not installed: {e}")
    print("Please install with:")
    print("pip install arxiv matplotlib pandas seaborn wordcloud openpyxl xlsxwriter")
    exit(1)

# Japanese font settings (optional)
plt.rcParams['font.family'] = ['DejaVu Sans']

class ArxivRoboticsAnalyzer:
    def __init__(self, keywords=None):
        self.papers = []

        def _load_config(file_path):
            with open(file_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        
        self.config = _load_config("./config/keywords.yaml")

        self.search_terms = self.config.get("keywords", [])
        print (self.search_terms)

    def search_arxiv_api(self, query, max_results=1000, start_year=2020):
        """Search papers using ArXiv API"""
        papers = []
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            for paper in search.results():
                # only include papers published from start_year
                if paper.published.year >= start_year:
                    papers.append({
                        'title': paper.title,
                        'authors': [str(author) for author in paper.authors],
                        'published': paper.published,
                        'updated': paper.updated,
                        'summary': paper.summary,
                        'categories': paper.categories,
                        'arxiv_id': paper.entry_id.split('/')[-1],
                        'pdf_url': paper.pdf_url
                    })
                    
        except Exception as e:
            print(f"Search error: {query} - {e}")
            
        return papers
    
    def search_all_terms(self, start_year=2020):
        """Collect papers for all search terms"""
        all_papers = []
        
        for category, terms in self.search_terms.items():
            print(f"\n=== Searching category: {category} ===")
            
            for term in terms:
                print(f"Searching term: {term}")
                query = f'all:"{term}" AND (cat:cs.RO OR cat:cs.AI OR cat:cs.CV OR cat:cs.LG)' # all: すべてのフィールドを検索
                papers = self.search_arxiv_api(query, max_results=500, start_year=start_year)
                
                for paper in papers:
                    paper['search_term'] = term
                    paper['category'] = category
                    
                all_papers.extend(papers)
                time.sleep(1)  # API rate limit countermeasure
                
        # Remove duplicates
        unique_papers = {}
        for paper in all_papers:
            arxiv_id = paper['arxiv_id']
            if arxiv_id not in unique_papers:
                unique_papers[arxiv_id] = paper
            else:
                # Retain the paper that matches more search terms
                existing = unique_papers[arxiv_id]
                if paper.get('relevance_score', 0) > existing.get('relevance_score', 0):
                    unique_papers[arxiv_id] = paper
        
        self.papers = list(unique_papers.values())
        print(f"\nTotal {len(self.papers)} papers retrieved")
        return self.papers
    
    def get_citation_count_semantic_scholar(self, arxiv_id):
        """Get citation count from Semantic Scholar API"""
        try:
            # Semantic Scholar API endpoint
            url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
            params = {
                'fields': 'citationCount,influentialCitationCount,year,title'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'citation_count': data.get('citationCount', 0),
                    'influential_citation_count': data.get('influentialCitationCount', 0),
                    'semantic_scholar_found': True
                }
            else:
                return {
                    'citation_count': 0,
                    'influential_citation_count': 0,
                    'semantic_scholar_found': False
                }
                
        except Exception as e:
            print(f"Citation lookup error for {arxiv_id}: {e}")
            return {
                'citation_count': 0,
                'influential_citation_count': 0,
                'semantic_scholar_found': False
            }
    
    def fetch_citation_counts(self):
        """Fetch citation counts for all papers"""
        print("Fetching citation counts from Semantic Scholar...")
        
        with tqdm(self.papers, desc="📚 Fetching citations", unit="papers", ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            for i, paper in enumerate(pbar, 1):
                citation_data = self.get_citation_count_semantic_scholar(paper['arxiv_id'])
                paper.update(citation_data)
                pbar.set_postfix_str(f"Paper: {paper['arxiv_id']}", refresh=True)
                time.sleep(0.05)  # 50ms delay between requests

        print("Citation count fetching completed.")
    
    def create_dataframe(self):
        """Create pandas DataFrame from papers"""
        if not self.papers:
            print("No paper data available")
            return None
            
        df_data = []
        for paper in self.papers:
            df_data.append({
                'arxiv_id': paper['arxiv_id'],
                'title': paper['title'],
                'authors': ', '.join(paper['authors'][:3]),  # First 3 authors
                'published_date': paper['published'],
                'year': paper['published'].year,
                'month': paper['published'].month,
                'categories': ', '.join(paper['categories']),
                'search_term': paper.get('search_term', ''),
                'category': paper.get('category', ''),
                'relevance_score': paper.get('relevance_score', 0),
                'abstract_length': len(paper['summary']),
                'num_authors': len(paper['authors']),
                'citation_count': paper.get('citation_count', 0),
                'influential_citation_count': paper.get('influential_citation_count', 0),
                'semantic_scholar_found': paper.get('semantic_scholar_found', False)
            })
        
        self.df = pd.DataFrame(df_data)
        return self.df
    
    def display_paper_list(self, sort_by='published_date', ascending=False, top_n=50, show_citations=True):
        """Display paper list in formatted table"""
        if self.df is None:
            print("No data available. Please run search first.")
            return
        
        # Sort dataframe
        df_sorted = self.df.sort_values(by=sort_by, ascending=ascending).head(top_n)
        
        print(f"\n=== Top {top_n} Papers (sorted by {sort_by}) ===")
        print("=" * 120)

        '''TBD
        for idx, paper in df_sorted.iterrows():
            print(f"\n[{paper['arxiv_id']}] {paper['title'][:80]}...")
            print(f"Authors: {paper['authors']}")
            print(f"Published: {paper['published_date'].strftime('%Y-%m-%d')} | Category: {paper['category']} | Term: {paper['search_term']}")
            
            if show_citations and paper['semantic_scholar_found']:
                print(f"Citations: {paper['citation_count']} (Influential: {paper['influential_citation_count']})")
            elif show_citations:
                print("Citations: Not available")
            
            print(f"ArXiv URL: https://arxiv.org/abs/{paper['arxiv_id']}")
            print("-" * 120)
        '''
    
    def display_top_cited_papers(self, top_n=20):
        """Display most cited papers"""
        if self.df is None:
            return
            
        # Filter papers with citation data
        cited_papers = self.df[self.df['semantic_scholar_found'] == True]
        
        if cited_papers.empty:
            print("No citation data available.")
            return
        
        # Sort by citation count
        top_cited = cited_papers.sort_values('citation_count', ascending=False).head(top_n)
        
        print(f"\n=== Top {top_n} Most Cited Papers ===")
        print("=" * 120)
        
        for idx, paper in top_cited.iterrows():
            print(f"\n{idx+1}. [{paper['arxiv_id']}] {paper['title'][:70]}...")
            print(f"Authors: {paper['authors']}")
            print(f"Published: {paper['published_date'].strftime('%Y-%m-%d')} | Citations: {paper['citation_count']} (Influential: {paper['influential_citation_count']})")
            print(f"Category: {paper['category']} | Search Term: {paper['search_term']}")
            print("-" * 120)
    
    def plot_publication_trends(self, save_path=None, csv_path=None, tikz_path=None):
        """Plot quarterly publication trends and export data"""
        if self.df is None:
            return
            
        # Create quarter-year column for better x-axis labels
        self.df['quarter'] = self.df['published_date'].dt.quarter
        self.df['year_quarter'] = self.df['year'].astype(str) + '-Q' + self.df['quarter'].astype(str)
        
        # Group by year-quarter
        quarterly_counts = self.df.groupby('year_quarter').size().sort_index()
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bar plot
        bars = ax.bar(range(len(quarterly_counts)), quarterly_counts.values, 
                     color='steelblue', alpha=0.8, edgecolor='darkblue', linewidth=1.5)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
        
        # Set x-axis labels
        ax.set_xticks(range(len(quarterly_counts)))
        ax.set_xticklabels(quarterly_counts.index, rotation=45, ha='right')
        
        # Styling
        ax.set_title('Quarterly Publication Trends', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Quarter', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Papers', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {save_path}")
        
        plt.show()
        
        # Export quarterly data to CSV
        if csv_path:
            quarterly_df = pd.DataFrame({
                'Quarter': quarterly_counts.index,
                'Papers': quarterly_counts.values
            })
            quarterly_df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"✅ Quarterly data exported to {csv_path}")
        
        # Export TikZ code
        if tikz_path:
            self._export_tikz(quarterly_counts, tikz_path)
            print(f"✅ TikZ code exported to {tikz_path}")
    
    def _export_tikz(self, quarterly_counts, tikz_path):
        """Export quarterly data as TikZ code for LaTeX"""
        tikz_code = []
        tikz_code.append("\\begin{tikzpicture}")
        tikz_code.append("\\begin{axis}[")
        tikz_code.append("    xlabel={Quarter},")
        tikz_code.append("    ylabel={Number of Papers},")
        tikz_code.append("    title={Quarterly Publication Trends},")
        tikz_code.append("    ybar,")
        tikz_code.append("    bar width=0.8,")
        tikz_code.append("    width=12cm,")
        tikz_code.append("    height=6cm,")
        tikz_code.append("    xtick=data,")
        tikz_code.append("    xticklabel style={rotate=45, anchor=east},")
        tikz_code.append("    nodes near coords,")
        tikz_code.append("    grid=major,")
        tikz_code.append("    grid style={dashed,gray!30},")
        tikz_code.append("]")
        
        # Add data
        tikz_code.append("\\addplot coordinates {")
        for i, (quarter, count) in enumerate(quarterly_counts.items()):
            tikz_code.append(f"    ({i},{count})")
        tikz_code.append("};")
        
        # Add x-tick labels
        tikz_code.append("\\pgfplotsset{")
        tikz_code.append("    xticklabels={")
        for i, quarter in enumerate(quarterly_counts.index):
            tikz_code.append(f"        {quarter},")
        tikz_code.append("    }")
        tikz_code.append("}")
        
        tikz_code.append("\\end{axis}")
        tikz_code.append("\\end{tikzpicture}")
        
        # Write to file
        with open(tikz_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(tikz_code))
        
        return '\n'.join(tikz_code)

    def create_citation_analysis_plot(self, save_path=None):
        """Create detailed citation analysis plots"""
        if self.df is None or 'citation_count' not in self.df.columns:
            return
            
        cited_papers = self.df[self.df['semantic_scholar_found'] == True]
        if cited_papers.empty:
            print("No citation data available for detailed analysis.")
            return
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Citation vs Time scatter with regression
        ax1 = axes[0, 0]
        x = cited_papers['year']
        y = cited_papers['citation_count']
        scatter = ax1.scatter(x, y, c=y, cmap='plasma', s=60, alpha=0.7, edgecolors='black')
        
        # Add regression line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax1.plot(x, p(x), "r--", alpha=0.8, linewidth=2)
        
        plt.colorbar(scatter, ax=ax1, label='Citations')
        ax1.set_title('📊 Citation Count vs Publication Year', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Publication Year')
        ax1.set_ylabel('Citation Count')
        
        # 2. Top cited papers
        ax2 = axes[0, 1]
        top_cited = cited_papers.nlargest(10, 'citation_count')
        bars = ax2.barh(range(len(top_cited)), top_cited['citation_count'], 
                       color=plt.cm.viridis(np.linspace(0, 1, len(top_cited))))
        ax2.set_yticks(range(len(top_cited)))
        ax2.set_yticklabels([f"{title[:30]}..." for title in top_cited['title']], fontsize=10)
        ax2.set_title('🏆 Top 10 Most Cited Papers', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Citation Count')
        
        # 3. Citation distribution by category
        ax3 = axes[0, 2]
        category_citations = cited_papers.groupby('category')['citation_count'].agg(['mean', 'std', 'count'])
        bars = ax3.bar(category_citations.index, category_citations['mean'], 
                      yerr=category_citations['std'], capsize=5, alpha=0.7)
        ax3.set_title('📈 Average Citations by Category', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Average Citation Count')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Citation impact over time
        ax4 = axes[1, 0]
        yearly_citations = cited_papers.groupby('year')['citation_count'].agg(['sum', 'mean'])
        ax4_twin = ax4.twinx()
        
        line1 = ax4.plot(yearly_citations.index, yearly_citations['sum'], 'b-o', linewidth=2, label='Total Citations')
        line2 = ax4_twin.plot(yearly_citations.index, yearly_citations['mean'], 'r-s', linewidth=2, label='Average Citations')
        
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Total Citations', color='blue')
        ax4_twin.set_ylabel('Average Citations', color='red')
        ax4.set_title('📅 Citation Impact Over Time', fontsize=14, fontweight='bold')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper left')
        
        # 5. Highly cited vs regular papers
        ax5 = axes[1, 1]
        high_cited = cited_papers[cited_papers['citation_count'] >= cited_papers['citation_count'].quantile(0.8)]
        regular_cited = cited_papers[cited_papers['citation_count'] < cited_papers['citation_count'].quantile(0.8)]
        
        categories = ['High Impact\n(Top 20%)', 'Regular Impact\n(Bottom 80%)']
        counts = [len(high_cited), len(regular_cited)]
        colors = ['gold', 'lightblue']
        
        wedges, texts, autotexts = ax5.pie(counts, labels=categories, autopct='%1.1f%%', 
                                          colors=colors, explode=(0.1, 0), shadow=True)
        ax5.set_title('🎯 Citation Impact Distribution', fontsize=14, fontweight='bold')
        
        # 6. Citation efficiency (citations per year since publication)
        ax6 = axes[1, 2]
        current_year = datetime.now().year
        cited_papers['years_since_pub'] = current_year - cited_papers['year']
        cited_papers['citation_efficiency'] = cited_papers['citation_count'] / np.maximum(cited_papers['years_since_pub'], 1)
        
        efficiency_data = cited_papers['citation_efficiency'][cited_papers['citation_efficiency'] > 0]
        ax6.hist(efficiency_data, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax6.axvline(efficiency_data.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {efficiency_data.mean():.1f}')
        ax6.set_title('⚡ Citation Efficiency\n(Citations per Year)', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Citations per Year')
        ax6.set_ylabel('Number of Papers')
        ax6.legend()
        
        plt.tight_layout()
        fig.suptitle('📚 Detailed Citation Analysis', fontsize=18, fontweight='bold', y=0.98)
        
        # if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def export_data(self, csv_path='./results/arxiv_robotics_papers.csv', excel_path=None):
        """Export paper data to CSV and optionally Excel format"""
        if self.df is None or self.df.empty:
            print("No data available for export.")
            return
        
        try:
            # Export to CSV
            export_df = self.df.copy()
            
            # Add summary field for export
            summaries = []
            for paper in self.papers:
                summaries.append(paper.get('summary', ''))
            
            export_df['summary'] = summaries[:len(export_df)]
            export_df['pdf_url'] = [f"https://arxiv.org/pdf/{arxiv_id}.pdf" for arxiv_id in export_df['arxiv_id']]
            export_df['arxiv_url'] = [f"https://arxiv.org/abs/{arxiv_id}" for arxiv_id in export_df['arxiv_id']]
            
            # Sort by publication date (newest first)
            export_df = export_df.sort_values('published_date', ascending=False)
            
            # Export to CSV
            export_df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"✅ Data exported to {csv_path} ({len(export_df)} papers)")
            
            # Export to Excel if path provided
            if excel_path:
                try:
                    # Convert timezone-aware datetime to timezone-naive for Excel
                    excel_df = export_df.copy()
                    excel_df['published_date'] = excel_df['published_date'].dt.tz_convert(None)
                    
                    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                        excel_df.to_excel(writer, sheet_name='Papers', index=False)
                        
                        # Create summary sheet
                        summary_data = {
                            'Metric': [
                                'Total Papers',
                                'Date Range',
                                'Categories',
                                'Average Citations',
                                'Most Cited Paper',
                                'Most Recent Paper'
                            ],
                            'Value': [
                                len(excel_df),
                                f"{excel_df['published_date'].min().strftime('%Y-%m-%d')} to {excel_df['published_date'].max().strftime('%Y-%m-%d')}",
                                len(excel_df['category'].unique()),
                                f"{excel_df['citation_count'].mean():.1f}",
                                excel_df.loc[excel_df['citation_count'].idxmax(), 'title'][:50] + "..." if len(excel_df) > 0 else "N/A",
                                excel_df.loc[excel_df['published_date'].idxmax(), 'title'][:50] + "..." if len(excel_df) > 0 else "N/A"
                            ]
                        }
                        summary_df = pd.DataFrame(summary_data)
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    print(f"✅ Data exported to {excel_path}")
                except ImportError:
                    print("⚠️ openpyxl not available for Excel export. Install with: pip install openpyxl")
                except Exception as e:
                    print(f"❌ Excel export failed: {e}")
            
        except Exception as e:
            print(f"❌ CSV export failed: {e}")
            return False
        
        return True
