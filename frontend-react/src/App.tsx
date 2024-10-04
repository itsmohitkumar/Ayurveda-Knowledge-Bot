import { useState } from 'react';

// Hardcoded variables
const API_URL = 'http://localhost:8000/ask'; // FastAPI endpoint
const HEADER_TITLE = 'Ayurveda-Knowledge-Bot';
const FOOTER_TEXT = '© 2024 Ayurveda-Knowledge-Bot';
const GITHUB_URL = 'https://github.com/your-github-repo';
const INPUT_PLACEHOLDER = 'Type your question here';
const ASK_BUTTON_TEXT = 'Ask';
const CLEAR_BUTTON_TEXT = 'Clear';
const LOADING_TEXT = 'Loading...';
const ERROR_TEXT = 'An error occurred';
const ASK_QUESTION_TITLE = 'Ask a question about Ayurveda';

// Interfaces
interface Question {
  question: string;
  answer: string;
}

export default function App() {
  const [questions, setQuestions] = useState<Question[]>([]);
  const [newQuestion, setNewQuestion] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleAsk = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(API_URL, {  // Using hardcoded API_URL
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: newQuestion }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch answer from the server.');
      }

      const data = await response.json();
      setQuestions([...questions, { question: newQuestion, answer: data.answer }]);
      setNewQuestion('');
    } catch (error) {
      setError(error instanceof Error ? error.message : ERROR_TEXT); // Using hardcoded ERROR_TEXT
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setQuestions([]);
    setNewQuestion('');
    setError(null);
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleAsk();
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-900">
      <header className="bg-purple-500 p-4 text-white shadow-md flex justify-between items-center">
        <img src="https://aws.amazon.com/favicon.ico" alt="AWS Logo" className="w-10 h-10" />
        <h1 className="text-4xl font-bold text-center">{HEADER_TITLE}</h1> {/* Using hardcoded HEADER_TITLE */}
        <img src="https://pdfjs.express/static/favicon.ico" alt="PDF.js Express Logo" className="w-10 h-10" />
      </header>
      <main className="flex-1 p-4 flex justify-center items-center">
        <div className="max-w-2xl w-full bg-gray-800 p-8 rounded-lg shadow-md">
          <div className="flex flex-col space-y-6">
            <h2 className="text-3xl font-bold text-purple-500">{ASK_QUESTION_TITLE}</h2> {/* Using hardcoded ASK_QUESTION_TITLE */}
            <div className="flex space-x-2">
              <input 
                type="text" 
                value={newQuestion} 
                onChange={(e) => setNewQuestion(e.target.value)} 
                onKeyPress={handleKeyPress} 
                placeholder={INPUT_PLACEHOLDER} // Using hardcoded INPUT_PLACEHOLDER
                className="w-full bg-gray-700 text-white p-3 rounded text-lg" 
                aria-label="Question input" 
              />
              <button 
                onClick={handleAsk} 
                className="bg-purple-500 hover:bg-purple-700 text-white p-3 rounded text-lg"
              >
                {ASK_BUTTON_TEXT} {/* Using hardcoded ASK_BUTTON_TEXT */}
              </button>
            </div>
            {loading ? (
              <p className="text-white text-lg">{LOADING_TEXT}</p>
            ) : error ? (
              <p className="text-red-500 text-lg">{error}</p>
            ) : (
              <div className="flex flex-col space-y-4">
                {questions.map((q, index) => (
                  <div key={index} className="bg-gray-700 p-4 rounded">
                    <p className="text-white font-bold text-lg">Question: {q.question}</p>
                    <p className="text-white text-lg">Answer: {q.answer}</p>
                  </div>
                ))}
              </div>
            )}
            <button 
              onClick={handleClear} 
              className="bg-gray-600 hover:bg-gray-500 text-white p-3 rounded text-lg"
            >
              {CLEAR_BUTTON_TEXT} {/* Using hardcoded CLEAR_BUTTON_TEXT */}
            </button>
          </div>
        </div>
      </main>
      <footer className="bg-purple-500 p-4 text-white shadow-md text-center">
        <p className="text-lg">{FOOTER_TEXT}</p> {/* Using hardcoded FOOTER_TEXT */}
        <p>
          <a 
            href={GITHUB_URL} // Using hardcoded GITHUB_URL
            target="_blank" 
            rel="noopener noreferrer" 
            className="text-yellow-300 hover:text-yellow-400 text-xl font-semibold transition duration-300"
          >
            View on GitHub
          </a>
        </p>
      </footer>
    </div>
  );
}
