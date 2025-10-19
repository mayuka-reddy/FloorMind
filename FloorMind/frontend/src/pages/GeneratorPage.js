import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Zap, Settings, Download, RefreshCw, Sparkles, Clock, Image as ImageIcon } from 'lucide-react';
import toast from 'react-hot-toast';

const GeneratorPage = () => {
  const [prompt, setPrompt] = useState('');
  const [modelType, setModelType] = useState('constraint_aware');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImage, setGeneratedImage] = useState(null);
  const [generationMetrics, setGenerationMetrics] = useState(null);
  const [include3D, setInclude3D] = useState(false);
  const [style, setStyle] = useState('modern');

  const samplePrompts = [
    "3-bedroom apartment with open kitchen and living room",
    "Small studio with bathroom and kitchenette",
    "2-story house with 4 bedrooms and 2 bathrooms",
    "Modern loft with master bedroom and walk-in closet",
    "Family home with garage and dining room",
    "Office space with conference room and reception area"
  ];

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      toast.error('Please enter a description for your floor plan');
      return;
    }

    setIsGenerating(true);
    
    try {
      // Simulate API call - replace with actual API integration
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      // Mock generated result
      const mockResult = {
        image_path: '/api/placeholder/512/512',
        generation_time: 2.3,
        metadata: {
          clip_score: 0.78,
          adjacency_score: 0.71,
          accuracy: 86.2
        }
      };
      
      setGeneratedImage(mockResult.image_path);
      setGenerationMetrics(mockResult);
      toast.success('Floor plan generated successfully!');
      
    } catch (error) {
      toast.error('Generation failed. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSamplePrompt = (samplePrompt) => {
    setPrompt(samplePrompt);
  };

  const handleDownload = () => {
    if (generatedImage) {
      toast.success('Download started!');
      // Implement actual download logic
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-secondary-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl lg:text-5xl font-bold text-gray-900 mb-4">
            <span className="gradient-text">AI Floor Plan</span> Generator
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Transform your ideas into detailed architectural floor plans using advanced AI models
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Panel */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            {/* Prompt Input */}
            <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
              <label className="block text-sm font-semibold text-gray-900 mb-3">
                Describe Your Floor Plan
              </label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter a detailed description of your desired floor plan..."
                className="w-full h-32 p-4 border border-gray-200 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
                disabled={isGenerating}
              />
              
              {/* Sample Prompts */}
              <div className="mt-4">
                <p className="text-sm font-medium text-gray-700 mb-2">Try these examples:</p>
                <div className="flex flex-wrap gap-2">
                  {samplePrompts.slice(0, 3).map((sample, index) => (
                    <button
                      key={index}
                      onClick={() => handleSamplePrompt(sample)}
                      className="text-xs px-3 py-1 bg-primary-50 text-primary-700 rounded-full hover:bg-primary-100 transition-colors"
                      disabled={isGenerating}
                    >
                      {sample}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Model Selection */}
            <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
              <label className="block text-sm font-semibold text-gray-900 mb-3">
                <Settings className="w-4 h-4 inline mr-2" />
                Model Selection
              </label>
              <div className="space-y-3">


                <label className="flex items-center space-x-3 cursor-pointer">
                  <input
                    type="radio"
                    name="model"
                    value="constraint_aware"
                    checked={modelType === 'constraint_aware'}
                    onChange={(e) => setModelType(e.target.value)}
                    className="text-primary-600 focus:ring-primary-500"
                    disabled={isGenerating}
                  />
                  <div className="flex-1">
                    <div className="font-medium text-gray-900">Constraint-Aware Model</div>
                    <div className="text-sm text-gray-600">
                      Advanced model with spatial consistency (Recommended)
                    </div>
                  </div>
                  <div className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded-full">
                    84.5% Accuracy
                  </div>
                </label>
                
                <label className="flex items-center space-x-3 cursor-pointer">
                  <input
                    type="radio"
                    name="model"
                    value="baseline"
                    checked={modelType === 'baseline'}
                    onChange={(e) => setModelType(e.target.value)}
                    className="text-primary-600 focus:ring-primary-500"
                    disabled={isGenerating}
                  />
                  <div className="flex-1">
                    <div className="font-medium text-gray-900">Baseline Model</div>
                    <div className="text-sm text-gray-600">
                      Standard Stable Diffusion fine-tuned model
                    </div>
                  </div>
                  <div className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full">
                    71.3% Accuracy
                  </div>
                </label>
              </div>
            </div>

            {/* Style and 3D Options */}
            <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
              <label className="block text-sm font-semibold text-gray-900 mb-3">
                <Sparkles className="w-4 h-4 inline mr-2" />
                Advanced Options
              </label>
              
              <div className="space-y-4">
                {/* Style Selection */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Architectural Style
                  </label>
                  <select
                    value={style}
                    onChange={(e) => setStyle(e.target.value)}
                    disabled={isGenerating}
                    className="w-full p-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  >
                    <option value="modern">Modern</option>
                    <option value="contemporary">Contemporary</option>
                    <option value="traditional">Traditional</option>
                    <option value="minimalist">Minimalist</option>
                    <option value="industrial">Industrial</option>
                    <option value="scandinavian">Scandinavian</option>
                  </select>
                </div>


              </div>
            </div>

            {/* Generate Button */}
            <motion.button
              onClick={handleGenerate}
              disabled={isGenerating || !prompt.trim()}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="w-full bg-gradient-to-r from-primary-600 to-secondary-600 text-white font-semibold py-4 px-6 rounded-xl hover:from-primary-700 hover:to-secondary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-xl disabled:hover:scale-100"
            >
              {isGenerating ? (
                <div className="flex items-center justify-center space-x-2">
                  <motion.div 
                    className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  />
                  <span>Generating Floor Plan...</span>
                </div>
              ) : (
                <div className="flex items-center justify-center space-x-2">
                  <Zap className="w-5 h-5" />
                  <span>Generate Floor Plan</span>
                </div>
              )}
            </motion.button>
          </motion.div>

          {/* Output Panel */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            {/* Generated Image */}
            <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">Generated Floor Plan</h3>
                {generatedImage && (
                  <button
                    onClick={handleDownload}
                    className="flex items-center space-x-2 px-4 py-2 bg-primary-100 text-primary-700 rounded-lg hover:bg-primary-200 transition-colors"
                  >
                    <Download className="w-4 h-4" />
                    <span>Download</span>
                  </button>
                )}
              </div>
              
              <div className="aspect-square bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl border-2 border-dashed border-gray-200 flex items-center justify-center relative overflow-hidden">
                {isGenerating ? (
                  <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-center"
                  >
                    <motion.div 
                      className="w-16 h-16 border-4 border-primary-200 border-t-primary-600 rounded-full mx-auto mb-4"
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    />
                    <motion.p 
                      className="text-gray-600 font-medium"
                      animate={{ opacity: [1, 0.5, 1] }}
                      transition={{ duration: 2, repeat: Infinity }}
                    >
                      Generating your floor plan...
                    </motion.p>
                    <p className="text-sm text-gray-500 mt-2">AI is analyzing spatial relationships</p>
                    
                    {/* Progress Animation */}
                    <div className="mt-4 w-48 mx-auto">
                      <div className="bg-gray-200 rounded-full h-2">
                        <motion.div 
                          className="bg-gradient-to-r from-primary-500 to-secondary-500 h-2 rounded-full"
                          initial={{ width: "0%" }}
                          animate={{ width: "100%" }}
                          transition={{ duration: 3, ease: "easeInOut" }}
                        />
                      </div>
                    </div>
                  </motion.div>
                ) : generatedImage ? (
                  <motion.div 
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="w-full h-full bg-gradient-to-br from-primary-50 to-secondary-50 rounded-lg flex items-center justify-center relative"
                  >
                    <div className="text-center text-gray-600">
                      <motion.div
                        animate={{ y: [-5, 5, -5] }}
                        transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                      >
                        <ImageIcon className="w-20 h-20 mx-auto mb-4 text-primary-500" />
                      </motion.div>
                      <p className="font-semibold text-lg">Generated Floor Plan</p>
                      <p className="text-sm text-primary-600 mt-1">Ready for download</p>
                    </div>
                    
                    {/* Success Animation */}
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: [0, 1.2, 1] }}
                      transition={{ delay: 0.5, duration: 0.5 }}
                      className="absolute top-4 right-4 w-8 h-8 bg-green-500 rounded-full flex items-center justify-center"
                    >
                      <span className="text-white text-sm">✓</span>
                    </motion.div>
                  </motion.div>
                ) : (
                  <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-center text-gray-400"
                  >
                    <motion.div
                      animate={{ scale: [1, 1.05, 1] }}
                      transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                    >
                      <ImageIcon className="w-20 h-20 mx-auto mb-4" />
                    </motion.div>
                    <p className="text-lg font-medium">Your floor plan will appear here</p>
                    <p className="text-sm mt-2">Enter a description and click generate to start</p>
                  </motion.div>
                )}
              </div>
            </div>

            {/* Generation Metrics */}
            {generationMetrics && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100"
              >
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Generation Metrics</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-4 bg-primary-50 rounded-lg">
                    <div className="text-2xl font-bold text-primary-600">
                      {generationMetrics.metadata.clip_score.toFixed(2)}
                    </div>
                    <div className="text-sm text-gray-600">CLIP Score</div>
                  </div>
                  <div className="text-center p-4 bg-secondary-50 rounded-lg">
                    <div className="text-2xl font-bold text-secondary-600">
                      {generationMetrics.metadata.adjacency_score.toFixed(2)}
                    </div>
                    <div className="text-sm text-gray-600">Adjacency</div>
                  </div>
                  <div className="text-center p-4 bg-accent-50 rounded-lg">
                    <div className="text-2xl font-bold text-accent-600">
                      {generationMetrics.metadata.accuracy.toFixed(1)}%
                    </div>
                    <div className="text-sm text-gray-600">Accuracy</div>
                  </div>
                  <div className="text-center p-4 bg-orange-50 rounded-lg">
                    <div className="text-2xl font-bold text-orange-600">
                      {generationMetrics.generation_time.toFixed(1)}s
                    </div>
                    <div className="text-sm text-gray-600">Gen Time</div>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Tips */}
            <div className="bg-gradient-to-r from-primary-50 to-secondary-50 rounded-2xl p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-3">
                <Sparkles className="w-5 h-5 inline mr-2" />
                Pro Tips
              </h3>
              <ul className="space-y-2 text-sm text-gray-700">
                <li>• Be specific about room types and their relationships</li>
                <li>• Mention desired adjacencies (e.g., "kitchen next to dining room")</li>
                <li>• Include approximate sizes or room counts for better results</li>
                <li>• Use architectural terms for more precise layouts</li>
              </ul>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default GeneratorPage;