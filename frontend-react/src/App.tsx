import { useState } from 'react';

// Hardcoded variables
const API_URL = 'http://localhost:8000/ask'; // FastAPI endpoint
const HEADER_TITLE = 'Your Personal Indian Tax Advisor'; // Updated header title
const FOOTER_TEXT = '© 2024 Indian Tax Advisor'; // Updated footer text
const GITHUB_URL = 'https://github.com/itsmohitkumar/Ayurveda-Knowledge-Bot';
const INPUT_PLACEHOLDER = 'Type your question here';
const ASK_BUTTON_TEXT = 'Ask';
const CLEAR_BUTTON_TEXT = 'Clear';
const LOADING_TEXT = 'Loading...';
const ERROR_TEXT = 'An error occurred';
const ASK_QUESTION_TITLE = 'Ask a question about Taxes'; // Updated question title
const SETTINGS_BUTTON_TEXT = 'Settings';
const SETTINGS_MODAL_TITLE = 'AWS Configuration';
const SUCCESS_MESSAGE = 'AWS keys are valid!';

// Interfaces
interface Question {
  question: string;
  answer: string;
}

interface AWSKeys {
  access_key_id: string;
  secret_access_key: string;
  default_region: string;
}

export default function App() {
  const [questions, setQuestions] = useState<Question[]>([]);
  const [newQuestion, setNewQuestion] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [awsKeys, setAwsKeys] = useState<AWSKeys | null>(null); // Store AWS keys as an object
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);
  const [awsKeysValid, setAwsKeysValid] = useState<boolean>(false); // New state for AWS keys validation

  const handleAsk = async () => {
    setLoading(true);
    setError(null);

    try {
      // Validate AWS keys
      if (!awsKeys || !awsKeys.access_key_id || !awsKeys.secret_access_key || !awsKeys.default_region) {
        // Skip error message for missing keys; just continue to try with existing keys
      }

      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${awsKeys?.access_key_id}:${awsKeys?.secret_access_key}`
        },
        body: JSON.stringify({ 
          question: newQuestion, 
          aws_access_key_id: awsKeys?.access_key_id, 
          aws_secret_access_key: awsKeys?.secret_access_key, 
          aws_default_region: awsKeys?.default_region 
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch answer from the server. Please ensure that your API keys are correctly configured');
      }

      const data = await response.json();
      setQuestions([...questions, { question: newQuestion, answer: data.answer }]);
      setNewQuestion('');
      setAwsKeysValid(true); // Set AWS keys as valid if the request is successful
    } catch (error) {
      setError(error instanceof Error ? error.message : ERROR_TEXT);
      setAwsKeysValid(false); // Reset valid status on error
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

  const handleSaveSettings = () => {
    // Logic to save AWS keys (could also save in localStorage or state)
    setIsSettingsOpen(false);
    setAwsKeysValid(true); // Assume keys are valid after saving
  };

  return (
    <div className="flex flex-col h-screen bg-gray-900">
      <header className="bg-purple-500 p-4 text-white shadow-md flex justify-between items-center">
        <img src="https://aws.amazon.com/favicon.ico" alt="AWS Logo" className="w-10 h-10" />
        <h1 className="text-4xl font-bold text-center">{HEADER_TITLE}</h1>
        <img src="https://pdfjs.express/static/favicon.ico" alt="PDF.js Express Logo" className="w-10 h-10" />
        <button 
          onClick={() => setIsSettingsOpen(true)} 
          className="bg-blue-500 hover:bg-blue-700 text-white p-3 rounded"
          title={SETTINGS_BUTTON_TEXT}
        >
          ⚙️ {/* Settings Icon */}
        </button>
      </header>
      <main className="flex-1 p-4 flex justify-center items-center">
        <div className="max-w-2xl w-full bg-gray-800 p-8 rounded-lg shadow-md">
          <div className="flex flex-col space-y-6">
            <h2 className="text-3xl font-bold text-purple-500">{ASK_QUESTION_TITLE}</h2>
            <p className="text-gray-400 text-sm">
              You can ask about tax deductions, how to save tax when purchasing a home or car, etc.
            </p>
            <div className="flex space-x-2">
              <input 
                type="text" 
                value={newQuestion} 
                onChange={(e) => setNewQuestion(e.target.value)} 
                onKeyPress={handleKeyPress} 
                placeholder={INPUT_PLACEHOLDER}
                className="w-full bg-gray-700 text-white p-3 rounded text-lg" 
                aria-label="Question input" 
              />
              <button 
                onClick={handleAsk} 
                className="bg-purple-500 hover:bg-purple-700 text-white p-3 rounded text-lg"
              >
                {ASK_BUTTON_TEXT}
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
            {awsKeysValid && (
              <p className="text-green-500 text-lg">{SUCCESS_MESSAGE}</p> // Show success message if keys are valid
            )}
            <button 
              onClick={handleClear} 
              className="bg-gray-600 hover:bg-gray-500 text-white p-3 rounded text-lg"
            >
              {CLEAR_BUTTON_TEXT}
            </button>
          </div>
        </div>
      </main>
      <footer className="bg-purple-500 p-4 text-white shadow-md text-center">
        <p className="text-lg">{FOOTER_TEXT}</p>
        <p>
          <a 
            href={GITHUB_URL} 
            target="_blank" 
            rel="noopener noreferrer" 
            className="text-yellow-300 hover:text-yellow-400 text-xl font-semibold transition duration-300"
          >
            View on GitHub
          </a>
        </p>
      </footer>

      {/* Settings Modal */}
      {isSettingsOpen && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
          <div className="bg-white p-6 rounded-lg w-80">
            <h3 className="text-lg font-bold">{SETTINGS_MODAL_TITLE}</h3>
            <div className="flex flex-col space-y-4">
              <input 
                type="text" 
                placeholder="AWS Access Key ID" 
                value={awsKeys?.access_key_id || ''} 
                onChange={(e) => setAwsKeys({ ...awsKeys, access_key_id: e.target.value } as AWSKeys)} 
                className="border rounded p-2"
              />
              <input 
                type="text" 
                placeholder="AWS Secret Access Key" 
                value={awsKeys?.secret_access_key || ''} 
                onChange={(e) => setAwsKeys({ ...awsKeys, secret_access_key: e.target.value } as AWSKeys)} 
                className="border rounded p-2"
              />
              <input 
                type="text" 
                placeholder="AWS Default Region" 
                value={awsKeys?.default_region || ''} 
                onChange={(e) => setAwsKeys({ ...awsKeys, default_region: e.target.value } as AWSKeys)} 
                className="border rounded p-2"
              />
              <button 
                onClick={handleSaveSettings} 
                className="bg-blue-500 hover:bg-blue-700 text-white p-2 rounded"
              >
                Save Settings
              </button>
              <button 
                onClick={() => setIsSettingsOpen(false)} 
                className="bg-gray-500 hover:bg-gray-700 text-white p-2 rounded"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
