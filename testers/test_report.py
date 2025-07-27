# Create test_report.py
import datetime
from pathlib import Path

def generate_report():
    report = f"""
RAG System Test Report
Generated: {datetime.datetime.now()}

1. Configuration:
   - PDFs in data/pdfs: {len(list(Path('data/pdfs').glob('*.pdf')))}
   - Vector store exists: {Path('vectorstore').exists()}
   - LLM Provider: Check .env file

2. Test Results:
   - [ ] Configuration test passed
   - [ ] Document ingestion successful
   - [ ] Vector search working
   - [ ] LLM integration working
   - [ ] Edge cases handled properly
   - [ ] Performance acceptable

3. Manual Testing:
   - [ ] Can answer factual questions
   - [ ] Can summarize content
   - [ ] Can compare concepts
   - [ ] Provides source citations
   - [ ] Handles invalid queries

4. Performance Metrics:
   - Average search time: ___ seconds
   - Average response time: ___ seconds
   - Total chunks indexed: ___

5. Issues Found:
   - List any problems here

6. Next Steps:
   - List improvements needed
"""
    
    with open("test_report.txt", "w") as f:
        f.write(report)
    
    print("Test report generated: test_report.txt")

generate_report()