"""
Core-2 엑소좀 miRNA 칵테일 분석 보고서를 PDF로 변환

Markdown → PDF 변환기
"""

import os
from pathlib import Path
import subprocess

def markdown_to_pdf_weasyprint(md_file, output_pdf):
    """
    WeasyPrint를 사용한 PDF 변환 (고품질)
    """
    try:
        import markdown
        from weasyprint import HTML, CSS
        
        # Markdown 읽기
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Markdown → HTML
        html_content = markdown.markdown(
            md_content,
            extensions=[
                'markdown.extensions.tables',
                'markdown.extensions.fenced_code',
                'markdown.extensions.codehilite',
                'markdown.extensions.toc'
            ]
        )
        
        # HTML 템플릿 (스타일 포함)
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                @page {{
                    size: A4;
                    margin: 2cm;
                    @bottom-right {{
                        content: "Page " counter(page) " of " counter(pages);
                        font-size: 9pt;
                    }}
                }}
                
                body {{
                    font-family: 'Malgun Gothic', 'Arial', sans-serif;
                    font-size: 11pt;
                    line-height: 1.6;
                    color: #333;
                }}
                
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                    page-break-before: always;
                }}
                
                h1:first-of-type {{
                    page-break-before: avoid;
                }}
                
                h2 {{
                    color: #34495e;
                    border-bottom: 2px solid #95a5a6;
                    padding-bottom: 5px;
                    margin-top: 30px;
                }}
                
                h3 {{
                    color: #555;
                    margin-top: 20px;
                }}
                
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                    font-size: 10pt;
                }}
                
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                
                th {{
                    background-color: #3498db;
                    color: white;
                    font-weight: bold;
                }}
                
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                
                code {{
                    background-color: #f4f4f4;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-family: 'Courier New', monospace;
                    font-size: 9pt;
                }}
                
                pre {{
                    background-color: #f8f8f8;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    overflow-x: auto;
                    font-size: 9pt;
                    line-height: 1.4;
                }}
                
                blockquote {{
                    border-left: 4px solid #3498db;
                    padding-left: 15px;
                    margin: 15px 0;
                    color: #555;
                    font-style: italic;
                }}
                
                img {{
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 20px auto;
                }}
                
                .page-break {{
                    page-break-after: always;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # PDF 생성
        HTML(string=html_template).write_pdf(output_pdf)
        
        print(f"✅ PDF 생성 완료: {output_pdf}")
        return True
        
    except ImportError as e:
        print(f"❌ 라이브러리 오류: {e}")
        print("필요한 패키지 설치:")
        print("  pip install markdown weasyprint")
        return False
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False


def markdown_to_pdf_pandoc(md_file, output_pdf):
    """
    Pandoc을 사용한 PDF 변환 (추천!)
    """
    try:
        cmd = [
            'pandoc',
            md_file,
            '-o', output_pdf,
            '--pdf-engine=xelatex',
            '-V', 'geometry:margin=2cm',
            '-V', 'fontsize=11pt',
            '-V', 'documentclass=article',
            '-V', 'mainfont=Malgun Gothic',
            '--toc',  # Table of contents
            '--toc-depth=3',
            '--highlight-style=tango'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ PDF 생성 완료 (Pandoc): {output_pdf}")
            return True
        else:
            print(f"❌ Pandoc 오류: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ Pandoc이 설치되어 있지 않습니다.")
        print("설치: https://pandoc.org/installing.html")
        return False
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False


def markdown_to_pdf_simple(md_file, output_pdf):
    """
    간단한 PDF 변환 (fpdf2 사용)
    """
    try:
        from fpdf import FPDF
        import re
        
        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 15)
                self.cell(0, 10, 'Core-2 Exosome miRNA Cocktail Analysis Report', 0, 1, 'C')
                self.ln(5)
            
            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
            
            def chapter_title(self, title):
                self.set_font('Arial', 'B', 14)
                self.set_fill_color(52, 152, 219)
                self.set_text_color(255, 255, 255)
                self.cell(0, 10, title, 0, 1, 'L', 1)
                self.ln(4)
                self.set_text_color(0, 0, 0)
            
            def chapter_body(self, body):
                # 한글 지원 필요
                self.set_font('Arial', '', 11)
                self.multi_cell(0, 5, body)
                self.ln()
        
        # Markdown 읽기
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # PDF 생성
        pdf = PDF()
        pdf.add_page()
        
        # 간단한 파싱 (제한적)
        lines = content.split('\n')
        for line in lines:
            if line.startswith('# '):
                pdf.chapter_title(line[2:])
            elif line.startswith('## '):
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, line[3:], 0, 1)
                pdf.set_font('Arial', '', 11)
            elif line.strip():
                try:
                    # ASCII만 지원 (한글 깨짐)
                    ascii_line = line.encode('ascii', 'ignore').decode('ascii')
                    if ascii_line:
                        pdf.multi_cell(0, 5, ascii_line)
                except:
                    pass
        
        pdf.output(output_pdf)
        print(f"✅ PDF 생성 완료 (Simple): {output_pdf}")
        print("⚠️ 한글 표시 제한 - Pandoc 또는 WeasyPrint 권장")
        return True
        
    except ImportError:
        print("❌ fpdf2 미설치")
        print("설치: pip install fpdf2")
        return False
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False


def main():
    """메인 실행"""
    # 파일 경로
    brain_dir = Path(r"C:\Users\brook\.gemini\antigravity\brain\47510bd0-5533-4ab4-8920-7dbbbfb00fe3")
    md_file = brain_dir / "Core2_Exosome_Comprehensive_Analysis.md"
    
    # 출력 경로 (Downloads)
    downloads_dir = Path(r"C:\Users\brook\Downloads")
    output_pdf = downloads_dir / "Core2_Exosome_Analysis_Report.pdf"
    
    print("="*80)
    print("Core-2 엑소좀 분석 보고서 PDF 변환")
    print("="*80)
    print(f"\n입력 파일: {md_file}")
    print(f"출력 파일: {output_pdf}\n")
    
    # 변환 시도 (우선순위)
    success = False
    
    # 1. Pandoc 시도 (최고 품질)
    print("[1/3] Pandoc 변환 시도...")
    success = markdown_to_pdf_pandoc(str(md_file), str(output_pdf))
    
    # 2. WeasyPrint 시도 (고품질)
    if not success:
        print("\n[2/3] WeasyPrint 변환 시도...")
        success = markdown_to_pdf_weasyprint(str(md_file), str(output_pdf))
    
    # 3. Simple 변환 (기본)
    if not success:
        print("\n[3/3] Simple 변환 시도...")
        success = markdown_to_pdf_simple(str(md_file), str(output_pdf))
    
    if success:
        print("\n" + "="*80)
        print("✅ PDF 변환 성공!")
        print("="*80)
        print(f"저장 위치: {output_pdf}")
        print(f"파일 크기: {output_pdf.stat().st_size / 1024:.1f} KB" if output_pdf.exists() else "")
        
        # 파일 열기
        import subprocess
        subprocess.run(['start', str(output_pdf)], shell=True)
    else:
        print("\n" + "="*80)
        print("❌ 모든 변환 방법 실패")
        print("="*80)
        print("\n해결 방법:")
        print("1. Pandoc 설치 (권장):")
        print("   https://pandoc.org/installing.html")
        print("\n2. 또는 Python 패키지 설치:")
        print("   pip install markdown weasyprint")


if __name__ == "__main__":
    main()
