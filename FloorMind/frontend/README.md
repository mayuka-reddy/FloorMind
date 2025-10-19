# FloorMind Frontend

A modern, responsive React application for the FloorMind AI-powered text-to-floorplan generator.

## Features

- **Modern UI/UX**: Clean, responsive design with Tailwind CSS
- **Interactive Generator**: Real-time floor plan generation interface
- **Model Comparison**: Detailed model performance and architecture views
- **Metrics Dashboard**: Comprehensive performance analytics
- **Responsive Design**: Optimized for desktop, tablet, and mobile devices
- **Smooth Animations**: Framer Motion for enhanced user experience

## Technology Stack

- **React 18**: Modern React with hooks and functional components
- **Tailwind CSS**: Utility-first CSS framework for rapid styling
- **Framer Motion**: Smooth animations and transitions
- **Recharts**: Interactive charts and data visualization
- **React Router**: Client-side routing
- **Lucide React**: Beautiful, customizable icons
- **React Hot Toast**: Elegant toast notifications

## Getting Started

### Prerequisites

- Node.js 16+ and npm/yarn
- FloorMind backend API running (see ../backend/README.md)

### Installation

1. **Install dependencies**
   ```bash
   npm install
   ```

2. **Start development server**
   ```bash
   npm start
   ```

3. **Open browser**
   Navigate to `http://localhost:3000`

### Build for Production

```bash
npm run build
```

## Project Structure

```
src/
├── components/          # Reusable UI components
│   └── layout/         # Layout components (Navbar, Footer)
├── pages/              # Page components
│   ├── HomePage.js     # Landing page
│   ├── GeneratorPage.js # Floor plan generator
│   ├── ModelsPage.js   # Model comparison
│   ├── MetricsPage.js  # Performance metrics
│   └── AboutPage.js    # About page
├── App.js              # Main app component
├── index.js            # App entry point
└── index.css           # Global styles
```

## Key Components

### HomePage
- Hero section with call-to-action
- Feature highlights
- Performance statistics
- Model comparison overview

### GeneratorPage
- Text input for floor plan descriptions
- Model selection (Baseline vs Constraint-Aware)
- Real-time generation interface
- Results display with metrics

### ModelsPage
- Detailed model architecture information
- Performance comparison charts
- Technical specifications
- Training process overview

### MetricsPage
- Interactive performance charts
- Training progress visualization
- Detailed metrics tables
- Key insights and analysis

### AboutPage
- Project mission and vision
- Development timeline
- Team information
- Technology stack details

## Styling

The application uses Tailwind CSS with a custom design system:

- **Primary Colors**: Blue gradient (#0ea5e9 to #0284c7)
- **Secondary Colors**: Purple gradient (#d946ef to #c026d3)
- **Accent Colors**: Green (#22c55e) for success states
- **Typography**: Inter font family for clean readability
- **Animations**: Smooth transitions and micro-interactions

## API Integration

The frontend is designed to integrate with the FloorMind Flask API:

- `POST /api/generate` - Generate floor plans
- `GET /api/evaluate` - Get model metrics
- `POST /api/generate/batch` - Batch generation

## Responsive Design

- **Mobile First**: Optimized for mobile devices
- **Breakpoints**: sm (640px), md (768px), lg (1024px), xl (1280px)
- **Flexible Layouts**: CSS Grid and Flexbox for adaptive layouts
- **Touch Friendly**: Appropriate touch targets and interactions

## Performance Optimizations

- **Code Splitting**: Automatic route-based code splitting
- **Lazy Loading**: Components loaded on demand
- **Image Optimization**: Responsive images with proper sizing
- **Bundle Analysis**: Webpack bundle analyzer for optimization

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Contributing

1. Follow the existing code style and patterns
2. Use TypeScript for new components (optional)
3. Ensure responsive design across all breakpoints
4. Add proper error handling and loading states
5. Include appropriate animations and transitions

## Environment Variables

Create a `.env` file in the frontend directory:

```
REACT_APP_API_URL=http://localhost:5000
REACT_APP_VERSION=1.0.0
```

## Deployment

The application can be deployed to various platforms:

- **Vercel**: `vercel --prod`
- **Netlify**: `netlify deploy --prod --dir=build`
- **AWS S3**: Upload build folder to S3 bucket
- **Docker**: Use provided Dockerfile for containerization

## Future Enhancements

- Real-time collaboration features
- 3D visualization integration
- Mobile app development
- Advanced customization options
- Multi-language support