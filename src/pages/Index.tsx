
import React from 'react';
import { Link } from 'react-router-dom';

const Index = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-12">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900 mb-6">Agentic AI Research Assistant</h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            A fully autonomous, modular, multi-agent research system built for 
            knowledge discovery, code generation, validation, and structured reporting.
          </p>
          
          <div className="bg-white shadow rounded-lg p-8 mb-10">
            <div className="flex flex-col md:flex-row items-center justify-center gap-8">
              <div className="w-full md:w-1/2">
                <h2 className="text-2xl font-semibold mb-4">Powered by GPT-4o</h2>
                <p className="text-gray-700 mb-4">
                  Harness the power of advanced AI models for research, with 
                  fallback to open-source alternatives. This system processes your 
                  research queries through multiple specialized agents.
                </p>
                <div className="flex items-center justify-center md:justify-start">
                  <a 
                    href="https://github.com/yourusername/agentic-ai-research" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="px-6 py-3 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
                  >
                    GitHub Repository
                  </a>
                </div>
              </div>
              <div className="w-full md:w-1/2">
                <img 
                  src="/placeholder.svg"
                  alt="AI Research Assistant"
                  className="w-full h-64 object-cover rounded shadow"
                />
              </div>
            </div>
          </div>
          
          <h2 className="text-2xl font-semibold mb-6">Key Features</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-xl font-medium mb-2">Autonomous Research</h3>
              <p className="text-gray-600">
                Fully automated knowledge retrieval from web, PDFs, and images
              </p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-xl font-medium mb-2">Code Generation</h3>
              <p className="text-gray-600">
                Produce reusable Python code with validation and documentation
              </p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-xl font-medium mb-2">Structured Reports</h3>
              <p className="text-gray-600">
                Generate comprehensive reports with visualizations and citations
              </p>
            </div>
          </div>
          
          <h2 className="text-2xl font-semibold mb-6">Get Started</h2>
          <p className="text-lg text-gray-700 mb-6">
            The frontend for this application is being built with Streamlit and Python.
            Clone the repository and follow the setup instructions to start using the system.
          </p>
          
          <div className="bg-gray-100 p-6 rounded-lg shadow-inner">
            <code className="block text-left overflow-x-auto whitespace-pre text-sm">
              git clone https://github.com/yourusername/agentic-ai-research.git<br/>
              cd agentic-ai-research<br/>
              pip install -r requirements.txt<br/>
              python src/main.py
            </code>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
