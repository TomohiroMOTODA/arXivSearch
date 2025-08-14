from datetime import datetime
from collections import Counter
import re
import time
import requests

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
except ImportError as e:
    print(f"Required libraries are not installed: {e}")
    print("Please install with:")
    print("pip install arxiv matplotlib pandas seaborn wordcloud openpyxl xlsxwriter")
    exit(1)

class ArxivRoboticsAnalyzer:
    def __init__(self):
        self.papers = []
        self.search_terms = {
            'VLA': [
                'Vision-Language-Action Model',
                'Vision-Language-Action',
                'VLA',
            ],
            # 'Robotics Foundation Model': [
            #     'Robotics Foundation Model',
            #     'RMB',
            #     'Robotic Foundation Model',
            #     'Robot Foundation Model'
            # ],
            # 'Robotic Manipulation': [
            #     'Robotic Manipulation',
            #     'RM',
            #     'Robotics Manipulation',
            #     'Robot Manipulation'
            # ]
        }
    
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
                query = f'all:"{term}" AND (cat:cs.RO OR cat:cs.AI OR cat:cs.CV OR cat:cs.LG)' # all: すべてのフィールドを検索している．
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
        
        for paper in tqdm(self.papers, desc="📚 Fetching citations", unit="📄 paper"):
            citation_data = self.get_citation_count_semantic_scholar(paper['arxiv_id'])
            paper.update(citation_data)
            time.sleep(0.1)  # 100ms delay between requests
            
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
    
    def plot_publication_trends(self, save_path=None):
        """Plot publication trends over time with enhanced visualizations"""
        if self.df is None:
            return
            
        # Set up the plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(20, 16))
        
        # Color palette
        colors = plt.cm.Set3(np.linspace(0, 1, 12))
        
        # 1. Yearly publication trends with trend line
        ax1 = plt.subplot(3, 4, 1)
        yearly_counts = self.df.groupby('year').size()
        bars = ax1.bar(yearly_counts.index, yearly_counts.values, color=colors[0], alpha=0.8, edgecolor='darkblue', linewidth=1.5)
        
        # Add trend line
        z = np.polyfit(yearly_counts.index, yearly_counts.values, 1)
        p = np.poly1d(z)
        ax1.plot(yearly_counts.index, p(yearly_counts.index), "r--", alpha=0.8, linewidth=2, label=f'Trend (slope: {z[0]:.2f})')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
        
        ax1.set_title('📈 Yearly Paper Submission Trends', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Papers', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Category-wise publication with pie chart
        ax2 = plt.subplot(3, 4, 2)
        category_counts = self.df.groupby('category').size()
        wedges, texts, autotexts = ax2.pie(category_counts.values, labels=category_counts.index, 
                                          autopct='%1.1f%%', colors=colors[:len(category_counts)],
                                          explode=[0.05] * len(category_counts), shadow=True, startangle=90)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax2.set_title('🎯 Research Category Distribution', fontsize=16, fontweight='bold', pad=20)
        
        # 3. Monthly trends (heatmap for recent years)
        ax3 = plt.subplot(3, 4, 3)
        recent_df = self.df[self.df['year'] >= 2022]
        if not recent_df.empty:
            monthly_pivot = recent_df.groupby(['year', 'month']).size().unstack(fill_value=0)
            sns.heatmap(monthly_pivot, annot=True, fmt='d', cmap='YlOrRd', ax=ax3, 
                       cbar_kws={'label': 'Number of Papers'})
            ax3.set_title('🔥 Monthly Publication Heatmap (2022+)', fontsize=16, fontweight='bold', pad=20)
            ax3.set_xlabel('Month', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Year', fontsize=12, fontweight='bold')
        
        # 4. Authors count distribution
        ax4 = plt.subplot(3, 4, 4)
        author_counts = self.df['num_authors'].value_counts().sort_index()
        ax4.bar(author_counts.index, author_counts.values, color=colors[3], alpha=0.8, edgecolor='darkgreen')
        ax4.set_title('👥 Author Count Distribution', fontsize=16, fontweight='bold', pad=20)
        ax4.set_xlabel('Number of Authors', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Number of Papers', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Citation analysis if available
        if 'citation_count' in self.df.columns and self.df['citation_count'].sum() > 0:
            # Citation vs Year scatter plot
            ax5 = plt.subplot(3, 4, 5)
            cited_papers = self.df[self.df['citation_count'] > 0]
            scatter = ax5.scatter(cited_papers['year'], cited_papers['citation_count'], 
                                 c=cited_papers['citation_count'], cmap='viridis', 
                                 s=cited_papers['citation_count']*2 + 20, alpha=0.6, edgecolors='black')
            
            plt.colorbar(scatter, ax=ax5, label='Citation Count')
            ax5.set_title('📊 Citations vs Publication Year', fontsize=16, fontweight='bold', pad=20)
            ax5.set_xlabel('Publication Year', fontsize=12, fontweight='bold')
            ax5.set_ylabel('Citation Count', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3)
            
            # Citation distribution histogram
            ax6 = plt.subplot(3, 4, 6)
            citation_data = self.df['citation_count'][self.df['citation_count'] > 0]
            n, bins, patches = ax6.hist(citation_data, bins=20, alpha=0.8, edgecolor='black', color=colors[5])
            
            # Color bars by value
            for i, (patch, bin_val) in enumerate(zip(patches, bins[:-1])):
                patch.set_facecolor(plt.cm.viridis(bin_val / max(bins)))
            
            ax6.set_title('📈 Citation Count Distribution', fontsize=16, fontweight='bold', pad=20)
            ax6.set_xlabel('Citation Count', fontsize=12, fontweight='bold')
            ax6.set_ylabel('Number of Papers', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3)
            
            # Top cited papers by category
            ax7 = plt.subplot(3, 4, 7)
            category_citations = self.df.groupby('category')['citation_count'].mean().sort_values(ascending=True)
            bars = ax7.barh(category_citations.index, category_citations.values, color=colors[6], alpha=0.8)
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax7.annotate(f'{width:.1f}', xy=(width, bar.get_y() + bar.get_height() / 2),
                           xytext=(3, 0), textcoords="offset points", ha='left', va='center', fontweight='bold')
            
            ax7.set_title('🏆 Average Citations by Category', fontsize=16, fontweight='bold', pad=20)
            ax7.set_xlabel('Average Citation Count', fontsize=12, fontweight='bold')
            ax7.grid(True, alpha=0.3)
        
        # 6. Abstract length distribution
        ax8 = plt.subplot(3, 4, 8)
        abstract_lengths = self.df['abstract_length']
        ax8.hist(abstract_lengths, bins=30, alpha=0.8, color=colors[7], edgecolor='black')
        ax8.axvline(abstract_lengths.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {abstract_lengths.mean():.0f}')
        ax8.axvline(abstract_lengths.median(), color='orange', linestyle='--', linewidth=2, 
                   label=f'Median: {abstract_lengths.median():.0f}')
        ax8.set_title('📝 Abstract Length Distribution', fontsize=16, fontweight='bold', pad=20)
        ax8.set_xlabel('Abstract Length (characters)', fontsize=12, fontweight='bold')
        ax8.set_ylabel('Number of Papers', fontsize=12, fontweight='bold')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 7. Search term frequency
        ax9 = plt.subplot(3, 4, 9)
        term_counts = self.df['search_term'].value_counts()
        bars = ax9.bar(range(len(term_counts)), term_counts.values, color=colors[8], alpha=0.8)
        ax9.set_xticks(range(len(term_counts)))
        ax9.set_xticklabels(term_counts.index, rotation=45, ha='right')
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax9.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
        
        ax9.set_title('🔍 Papers by Search Term', fontsize=16, fontweight='bold', pad=20)
        ax9.set_ylabel('Number of Papers', fontsize=12, fontweight='bold')
        ax9.grid(True, alpha=0.3)
        
        # 8. Quarterly trends
        ax10 = plt.subplot(3, 4, 10)
        self.df['quarter'] = self.df['published_date'].dt.quarter
        quarterly_data = self.df.groupby(['year', 'quarter']).size().unstack(fill_value=0)
        
        if not quarterly_data.empty:
            quarterly_data.plot(kind='bar', stacked=True, ax=ax10, color=colors[9:13])
            ax10.set_title('📅 Quarterly Publication Trends', fontsize=16, fontweight='bold', pad=20)
            ax10.set_xlabel('Year', fontsize=12, fontweight='bold')
            ax10.set_ylabel('Number of Papers', fontsize=12, fontweight='bold')
            ax10.legend(title='Quarter', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax10.tick_params(axis='x', rotation=45)
        
        # 9. Category evolution over time
        ax11 = plt.subplot(3, 4, 11)
        category_yearly = self.df.groupby(['year', 'category']).size().unstack(fill_value=0)
        category_yearly.plot(kind='area', stacked=True, ax=ax11, alpha=0.7, color=colors[:len(category_yearly.columns)])
        ax11.set_title('🌊 Category Evolution Over Time', fontsize=16, fontweight='bold', pad=20)
        ax11.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax11.set_ylabel('Number of Papers', fontsize=12, fontweight='bold')
        ax11.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax11.grid(True, alpha=0.3)
        
        # 10. Recent activity (last 12 months)
        ax12 = plt.subplot(3, 4, 12)
        recent_12_months = self.df[self.df['published_date'] >= (pd.Timestamp.now(tz='UTC') - timedelta(days=365))]
        if not recent_12_months.empty:
            monthly_recent = recent_12_months.groupby(recent_12_months['published_date'].dt.to_period('M')).size()
            line = ax12.plot(monthly_recent.index.astype(str), monthly_recent.values, 
                           marker='o', linewidth=3, markersize=8, color=colors[1])
            ax12.fill_between(range(len(monthly_recent)), monthly_recent.values, alpha=0.3, color=colors[1])
            
            ax12.set_title('🚀 Recent Activity (Last 12 Months)', fontsize=16, fontweight='bold', pad=20)
            ax12.set_xlabel('Month', fontsize=12, fontweight='bold')
            ax12.set_ylabel('Number of Papers', fontsize=12, fontweight='bold')
            ax12.tick_params(axis='x', rotation=45)
            ax12.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout(pad=3.0)
        
        # Add main title
        fig.suptitle('🤖 ArXiv Robotics Research Analysis Dashboard', 
                    fontsize=24, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

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

    def export_data(self, csv_path='arxiv_robotics_papers.csv', excel_path=None):
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

def main():
    """Main execution function"""
    print("Starting ArXiv Robotics Foundation Model & VLA research trend analysis...")
    
    # Initialize analyzer
    analyzer = ArxivRoboticsAnalyzer()
    
    # Search papers (from 2020)
    print("Searching for papers...")
    papers = analyzer.search_all_terms(start_year=2020)
    
    if not papers:
        print("No papers found.")
        return
    
    # Fetch citation counts
    print("Fetching citation counts...")
    analyzer.fetch_citation_counts()
    
    # Create dataframe
    print("Organizing data...")
    df = analyzer.create_dataframe()
    
    # Display paper lists
    print("Displaying paper lists...")
    analyzer.display_paper_list(sort_by='published_date', top_n=30)
    analyzer.display_top_cited_papers(top_n=15)
    
    # Visualize results
    print("Generating enhanced plots...")
    analyzer.plot_publication_trends(save_path='arxiv_robotics_trends.png')
    analyzer.create_citation_analysis_plot(save_path='arxiv_citation_analysis.png')
    
    # Keyword analysis
    print("Analyzing keywords...")
    # top_keywords = analyzer.analyze_keywords(top_n=30)  # TODO: Implement this method
    
    # Report generation
    print("Generating summary report...")
    # analyzer.generate_summary_report(output_path='arxiv_robotics_analysis_report.md')  # TODO: Implement this method
    
    # Data export
    print("Exporting data...")
    analyzer.export_data(
        csv_path='arxiv_robotics_papers.csv',
        excel_path='arxiv_robotics_analysis.xlsx'
    )
    
    print("\nAnalysis completed!")
    print("Generated files:")
    print("- arxiv_robotics_trends.png (trend plot)")
    print("- arxiv_citation_analysis.png (citation analysis)")
    print("- arxiv_robotics_papers.csv (paper data)")

if __name__ == "__main__":
    # Japanese font settings (optional)
    plt.rcParams['font.family'] = ['DejaVu Sans']
    main()
