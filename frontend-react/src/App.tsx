import { useState } from 'react';

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
      const response = await fetch('/ask', {
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
      setError(error instanceof Error ? error.message : 'An error occurred');
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
        <img src="https://aws.amazon.com/favicon.ico" alt="AWS Logo" className="w-8 h-8" />
        <h1 className="text-3xl font-bold text-center">Ayurveda-Knowledge-Bot</h1>
        <img src="https://pdfjs.express/static/favicon.ico" alt="PDF.js Express Logo" className="w-8 h-8" />
      </header>
      <main className="flex-1 p-4 flex justify-center items-center">
        <div className="max-w-2xl w-full bg-gray-800 p-8 rounded-lg shadow-md">
          <div className="flex flex-col space-y-4">
            <h2 className="text-2xl font-bold text-purple-500">Ask a question about Ayurveda</h2>
            <div className="flex space-x-2">
              <input 
                type="text" 
                value={newQuestion} 
                onChange={(e) => setNewQuestion(e.target.value)} 
                onKeyPress={handleKeyPress} 
                placeholder="Type your question here" 
                className="w-full bg-gray-700 text-white p-2 rounded" 
                aria-label="Question input" 
              />
              <button 
                onClick={handleAsk} 
                className="bg-purple-500 hover:bg-purple-700 text-white p-2 rounded"
              >
                Ask
              </button>
            </div>
            {loading ? (
              <p className="text-white">Loading...</p>
            ) : error ? (
              <p className="text-red-500">{error}</p>
            ) : (
              <div className="flex flex-col space-y-4">
                {questions.map((q, index) => (
                  <div key={index} className="bg-gray-700 p-4 rounded">
                    <p className="text-white font-bold">Question: {q.question}</p>
                    <p className="text-white">Answer: {q.answer}</p>
                  </div>
                ))}
              </div>
            )}
            <button 
              onClick={handleClear} 
              className="bg-gray-600 hover:bg-gray-500 text-white p-2 rounded"
            >
              Clear
            </button>
          </div>
        </div>
      </main>
      <footer className="bg-purple-500 p-4 text-white shadow-md text-center">
        <p>&copy; 2024 Ayurveda-Knowledge-Bot</p>
        <p>
          <a href="https://github.com/your-github-repo" target="_blank" rel="noopener noreferrer" className="text-white hover:text-gray-200">
            View on GitHub
          </a>
        </p>
      </footer>
    </div>
  );
}
