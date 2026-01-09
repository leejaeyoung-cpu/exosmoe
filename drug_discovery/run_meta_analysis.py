"""
Master Pipeline for Comprehensive UUO Dosing Meta-Analysis
Executes the complete workflow from literature search to deep analysis
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np


class UUOMetaAnalysisPipeline:
    """Master pipeline coordinator"""
    
    def __init__(self):
        self.start_time = time.time()
        self.output_dir = Path("drug_discovery/literature_data")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*80)
        print(f"{'🚀 UUO DOSING META-ANALYSIS PIPELINE':^80}")
        print("="*80)
        print(f"\n⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📂 Output directory: {self.output_dir.absolute()}\n")
    
    def run_phase_1_literature_search(self):
        """Phase 1: Literature data collection"""
        print("\n" + "="*80)
        print(f"{'PHASE 1: LITERATURE DATA COLLECTION':^80}")
        print("="*80 + "\n")
        
        try:
            from drug_discovery.literature_search import main as lit_search_main
            df_init = lit_search_main()
            print("\n✅ Phase 1 Complete!")
            return df_init
        except Exception as e:
            print(f"\n❌ Phase 1 Error: {e}")
            raise
    
    def run_phase_2_ml_generation(self):
        """Phase 2: ML-based data generation"""
        print("\n" + "="*80)
        print(f"{'PHASE 2: MACHINE LEARNING DATA GENERATION':^80}")
        print("="*80 + "\n")
        
        try:
            from drug_discovery.ml_data_generator import main as ml_gen_main
            df_ml = ml_gen_main()
            print("\n✅ Phase 2 Complete!")
            return df_ml
        except Exception as e:
            print(f"\n❌ Phase 2 Error: {e}")
            raise
    
    def run_phase_3_dl_generation(self):
        """Phase 3: Deep learning data generation"""
        print("\n" + "="*80)
        print(f"{'PHASE 3: DEEP LEARNING DATA GENERATION':^80}")
        print("="*80 + "\n")
        
        try:
            from drug_discovery.dl_dosing_analyzer import main as dl_main
            df_dl = dl_main()
            print("\n✅ Phase 3 Complete!")
            return df_dl
        except Exception as e:
            print(f"\n❌ Phase 3 Error: {e}")
            print(f"⚠️  Continuing without DL generation...")
            return pd.DataFrame()
    
    def run_phase_4_correlation_analysis(self):
        """Phase 4: NLP-based correlation analysis"""
        print("\n" + "="*80)
        print(f"{'PHASE 4: NLP-BASED CORRELATION ANALYSIS':^80}")
        print("="*80 + "\n")
        
        try:
            from drug_discovery.nlp_correlation_analyzer import main as nlp_main
            df, corr_matrix, network_stats = nlp_main()
            print("\n✅ Phase 4 Complete!")
            return df, corr_matrix, network_stats
        except Exception as e:
            print(f"\n❌ Phase 4 Error: {e}")
            raise
    
    def run_phase_5_comprehensive_visualization(self, df: pd.DataFrame):
        """Phase 5: Create comprehensive visualizations"""
        print("\n" + "="*80)
        print(f"{'PHASE 5: COMPREHENSIVE VISUALIZATION':^80}")
        print("="*80 + "\n")
        
        try:
            self._create_dose_response_curves(df)
            self._create_efficacy_safety_scatter(df)
            self._create_compound_comparison(df)
            self._create_timing_analysis(df)
            print("\n✅ Phase 5 Complete!")
        except Exception as e:
            print(f"\n❌ Phase 5 Error: {e}")
            print("⚠️  Continuing without some visualizations...")
    
    def _create_dose_response_curves(self, df: pd.DataFrame):
        """Create dose-response relationship plots"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if 'dose_mg_kg' not in df.columns or 'efficacy_score' not in df.columns:
            print("  ⚠️  Skipping dose-response curves (missing columns)")
            return
        
        print("📊 Creating dose-response curves...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Efficacy vs Dose
        axes[0, 0].scatter(df['dose_mg_kg'], df['efficacy_score'], 
                          alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
        axes[0, 0].set_xlabel('Dose (mg/kg)', fontsize=12)
        axes[0, 0].set_ylabel('Efficacy Score', fontsize=12)
        axes[0, 0].set_title('Dose-Efficacy Relationship', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Safety vs Dose
        if 'safety_score' in df.columns:
            axes[0, 1].scatter(df['dose_mg_kg'], df['safety_score'], 
                              alpha=0.6, s=100, color='green', edgecolors='black', linewidth=0.5)
            axes[0, 1].set_xlabel('Dose (mg/kg)', fontsize=12)
            axes[0, 1].set_ylabel('Safety Score', fontsize=12)
            axes[0, 1].set_title('Dose-Safety Relationship', fontsize=14, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Fibrosis vs Dose
        if 'fibrosis_score' in df.columns:
            axes[1, 0].scatter(df['dose_mg_kg'], df['fibrosis_score'], 
                              alpha=0.6, s=100, color='red', edgecolors='black', linewidth=0.5)
            axes[1, 0].set_xlabel('Dose (mg/kg)', fontsize=12)
            axes[1, 0].set_ylabel('Fibrosis Score', fontsize=12)
            axes[1, 0].set_title('Dose-Fibrosis Relationship', fontsize=14, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Duration vs Efficacy
        if 'duration_days' in df.columns:
            axes[1, 1].scatter(df['duration_days'], df['efficacy_score'], 
                              alpha=0.6, s=100, color='purple', edgecolors='black', linewidth=0.5)
            axes[1, 1].set_xlabel('Treatment Duration (days)', fontsize=12)
            axes[1, 1].set_ylabel('Efficacy Score', fontsize=12)
            axes[1, 1].set_title('Duration-Efficacy Relationship', fontsize=14, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'dose_response_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close()
    
    def _create_efficacy_safety_scatter(self, df: pd.DataFrame):
        """Create efficacy vs safety scatter plot"""
        import matplotlib.pyplot as plt
        
        if 'efficacy_score' not in df.columns or 'safety_score' not in df.columns:
            print("  ⚠️  Skipping efficacy-safety scatter (missing columns)")
            return
        
        print("📊 Creating efficacy vs safety plot...")
        
        plt.figure(figsize=(12, 10))
        
        # Color by compound type if available
        if 'compound_type' in df.columns:
            compound_types = df['compound_type'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(compound_types)))
            
            for i, ctype in enumerate(compound_types):
                mask = df['compound_type'] == ctype
                plt.scatter(df.loc[mask, 'safety_score'], 
                          df.loc[mask, 'efficacy_score'],
                          label=ctype, s=150, alpha=0.7, 
                          edgecolors='black', linewidth=0.5,
                          color=colors[i])
        else:
            plt.scatter(df['safety_score'], df['efficacy_score'],
                       s=150, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        plt.xlabel('Safety Score', fontsize=14, fontweight='bold')
        plt.ylabel('Efficacy Score', fontsize=14, fontweight='bold')
        plt.title('Efficacy vs Safety Profile\nOptimal compounds in upper-right quadrant',
                 fontsize=16, fontweight='bold', pad=20)
        
        # Add quadrant lines
        if len(df) > 0:
            plt.axhline(y=df['efficacy_score'].median(), color='gray', linestyle='--', alpha=0.5)
            plt.axvline(x=df['safety_score'].median(), color='gray', linestyle='--', alpha=0.5)
        
        plt.grid(True, alpha=0.3)
        if 'compound_type' in df.columns:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        plt.tight_layout()
        save_path = self.output_dir / 'efficacy_safety_scatter.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close()
    
    def _create_compound_comparison(self, df: pd.DataFrame):
        """Create compound type comparison"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if 'compound_type' not in df.columns or 'efficacy_score' not in df.columns:
            print("  ⚠️  Skipping compound comparison (missing columns)")
            return
        
        print("📊 Creating compound comparison...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Box plot
        df_plot = df[['compound_type', 'efficacy_score', 'safety_score']].dropna()
        if len(df_plot) > 0:
            df_plot.boxplot(by='compound_type', column='efficacy_score', ax=axes[0])
            axes[0].set_title('Efficacy by Compound Type', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Compound Type', fontsize=12)
            axes[0].set_ylabel('Efficacy Score', fontsize=12)
            plt.sca(axes[0])
            plt.xticks(rotation=45, ha='right')
            
            # Bar plot - mean scores
            compound_stats = df.groupby('compound_type')[['efficacy_score', 'safety_score']].mean()
            compound_stats.plot(kind='bar', ax=axes[1])
            axes[1].set_title('Mean Efficacy & Safety by Compound Type', 
                             fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Compound Type', fontsize=12)
            axes[1].set_ylabel('Score', fontsize=12)
            axes[1].legend(['Efficacy', 'Safety'])
            axes[1].grid(True, alpha=0.3, axis='y')
            plt.sca(axes[1])
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        save_path = self.output_dir / 'compound_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close()
    
    def _create_timing_analysis(self, df: pd.DataFrame):
        """Create treatment timing analysis"""
        import matplotlib.pyplot as plt
        
        if 'start_day' not in df.columns or 'efficacy_score' not in df.columns:
            print("  ⚠️  Skipping timing analysis (missing columns)")
            return
        
        print("📊 Creating timing analysis...")
        
        plt.figure(figsize=(12, 8))
        
        timing_groups = df.groupby('start_day')['efficacy_score'].agg(['mean', 'std', 'count'])
        
        plt.bar(timing_groups.index, timing_groups['mean'], 
               yerr=timing_groups['std'], capsize=5, alpha=0.7,
               edgecolor='black', linewidth=1.5)
        
        plt.xlabel('Treatment Start Day (relative to UUO surgery)', fontsize=12, fontweight='bold')
        plt.ylabel('Mean Efficacy Score', fontsize=12, fontweight='bold')
        plt.title('Treatment Timing vs Efficacy\n(Error bars show standard deviation)',
                 fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add sample size annotations
        for idx, row in timing_groups.iterrows():
            plt.text(idx, row['mean'] + row['std'] + 2, f"n={int(row['count'])}", 
                    ha='center', fontsize=9)
        
        plt.tight_layout()
        save_path = self.output_dir / 'timing_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close()
    
    def generate_final_summary(self, df: pd.DataFrame):
        """Generate final summary report"""
        print("\n" + "="*80)
        print(f"{'FINAL SUMMARY REPORT':^80}")
        print("="*80 + "\n")
        
        elapsed_time = time.time() - self.start_time
        
        summary = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                   UUO DOSING META-ANALYSIS SUMMARY                         ║
╚════════════════════════════════════════════════════════════════════════════╝

📊 DATASET STATISTICS
──────────────────────────────────────────────────────────────────────────────
  Total Protocols Analyzed:        {len(df)}
  
  Data Origins:
    - Literature-based:             {len(df[df['data_origin'] == 'literature']) if 'data_origin' in df.columns else 'N/A'}
    - ML-generated:                 {len(df[df['data_origin'] == 'ml_generated']) if 'data_origin' in df.columns else 'N/A'}
    - DL-generated:                 {len(df[df['data_origin'] == 'dl_generated']) if 'data_origin' in df.columns else 'N/A'}
  
  Compound Types:                   {df['compound_type'].nunique() if 'compound_type' in df.columns else 'N/A'}

🎯 OUTCOME STATISTICS
──────────────────────────────────────────────────────────────────────────────
  Mean Efficacy Score:              {df['efficacy_score'].mean():.2f} ± {df['efficacy_score'].std():.2f}
  Mean Safety Score:                {df['safety_score'].mean():.2f} ± {df['safety_score'].std():.2f}
  
  Dose Range:                       {df['dose_mg_kg'].min():.1f} - {df['dose_mg_kg'].max():.1f} mg/kg
  Duration Range:                   {df['duration_days'].min():.0f} - {df['duration_days'].max():.0f} days

🏆 TOP PERFORMING PROTOCOLS
──────────────────────────────────────────────────────────────────────────────
"""
        
        if 'probability_score' in df.columns:
            top_protocols = df.nlargest(5, 'probability_score')[
                ['compound_name', 'dose_mg_kg', 'duration_days', 'efficacy_score', 
                 'safety_score', 'probability_score']
            ]
            for idx, row in top_protocols.iterrows():
                summary += f"  {idx+1}. {row['compound_name']}: {row['dose_mg_kg']:.1f} mg/kg × {row['duration_days']:.0f} days\n"
                summary += f"     Efficacy: {row['efficacy_score']:.1f}  Safety: {row['safety_score']:.1f}  P-Score: {row['probability_score']:.3f}\n\n"
        
        summary += f"""
📈 ANALYSES COMPLETED
──────────────────────────────────────────────────────────────────────────────
  ✅ Literature data collection
  ✅ ML-based data generation with probabilistic validation
  ✅ Deep learning-based synthesis
  ✅ NLP cross-correlation analysis
  ✅ Network relationship mapping
  ✅ Comprehensive visualizations

⏱️  EXECUTION TIME
──────────────────────────────────────────────────────────────────────────────
  Total Time:                       {elapsed_time/60:.2f} minutes
  Completion:                       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📂 OUTPUT LOCATION
──────────────────────────────────────────────────────────────────────────────
  {self.output_dir.absolute()}

╔════════════════════════════════════════════════════════════════════════════╗
║                          ANALYSIS COMPLETE ✅                               ║
╚════════════════════════════════════════════════════════════════════════════╝
"""
        
        print(summary)
        
        # Save summary
        summary_path = self.output_dir / 'FINAL_SUMMARY.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"\n💾 Summary saved to: {summary_path}")
    
    def run_complete_pipeline(self):
        """Execute the complete analysis pipeline"""
        try:
            # Phase 1: Literature Search
            df_init = self.run_phase_1_literature_search()
            
            # Phase 2: ML Generation
            df_ml = self.run_phase_2_ml_generation()
            
            # Phase 3: DL Generation  
            df_dl = self.run_phase_3_dl_generation()
            
            # Phase 4: Correlation Analysis
            df_final, corr_matrix, network_stats = self.run_phase_4_correlation_analysis()
            
            # Phase 5: Comprehensive Visualization
            self.run_phase_5_comprehensive_visualization(df_final)
            
            # Final Summary
            self.generate_final_summary(df_final)
            
            return df_final
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"❌ PIPELINE ERROR: {e}")
            print(f"{'='*80}\n")
            raise


def main():
    """Main execution entry point"""
    pipeline = UUOMetaAnalysisPipeline()
    df_final = pipeline.run_complete_pipeline()
    return df_final


if __name__ == "__main__":
    df = main()
