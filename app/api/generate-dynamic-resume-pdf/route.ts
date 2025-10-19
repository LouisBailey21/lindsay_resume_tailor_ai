import { NextRequest, NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import { OpenAI } from 'openai';
import { PDFDocument, StandardFonts, rgb } from 'pdf-lib';

// Helper to parse resume text
function parseResume(resumeText: string) {
  const lines = resumeText.split('\n');
  const info: string[] = [];
  let bodyStart = 0;
  for (let idx = 0; idx < lines.length; idx++) {
    if (lines[idx].trim()) info.push(lines[idx].trim());
    if (info.length === 6) {
      bodyStart = idx + 1;
      break;
    }
  }
  const [headline, name, email, phone, location, linkedin] = info;
  while (bodyStart < lines.length && !lines[bodyStart].trim()) bodyStart++;
  const body = lines.slice(bodyStart).join('\n');
  return { headline, name, email, phone, location, linkedin, body };
}

// Helper to build OpenAI prompt
function buildPrompt(baseResume: string, jobDescription: string) {
  return `
You are a world-class technical resume assistant.

SYSTEM INSTRUCTION: Make the resume align as closely as possible with the Job Description (JD). Must proactively REPLACE, REPHRASE, and ADD bullet points under each Experience entry, especially recent/current roles, to ensure the language, skills, and technologies match the JD specifically. Do NOT leave any Experience section or bullet point unchanged if it could better reflect or incorporate keywords, duties, or requirements from the JD. Acceptable and encouraged to write NEW bullet points where there are relevant experiences (even if not previously mentioned). Prioritize jobs/roles closest to the desired job.

Your main objectives:
1. Maximize keyword/skills and responsibilities match between the resume and the job description (JD). Use the exact relevant technology, tool, process, or methodology names from the JD wherever accurate.
2. Preserve all original company names, job titles, and periods/dates in the Professional Experience section.
3. In each Experience/job entry, ensure 6–8 highly relevant and impactful bullet points. Aggressively update, rewrite, or add new ones so they reflect the actual duties, skills, or stacks requested in the JD—especially prioritizing skills, tools, or requirements from the current and most recent positions. If an original bullet or responsibility does not closely match the JD, replace or revise it.
4. Make the experiences emphasize the main tech stack from the JD in the most recent or relevant roles, and distribute additional or secondary JD requirements across earlier positions naturally. Each company’s experience should collectively cover the full range of JD skills and duties.
5. Place the SKILLS section immediately after the SUMMARY section and before the PROFESSIONAL EXPERIENCE section. This ensures all key stacks and technologies are visible at the top of the resume for ATS and recruiters.
6. In the Summary, integrate the most essential and high-priority skills, stacks, and requirements from the JD, emphasizing the strongest elements from the original. Keep it dense with relevant keywords and technologies, but natural in tone.
7. In every section (Summary, Skills, Experience), INCLUDE as many relevant unique keywords and technologies from the job description as possible.
8. CRITICAL SKILLS SECTION: Create an EXCEPTIONALLY RICH, DENSE, and COMPREHENSIVE Skills section. Extract and list EVERY technology, tool, framework, library, service, and methodology from BOTH the JD AND candidate's experience. Make it so comprehensive it dominates keyword matching.
MANDATORY STRUCTURE (IN THIS EXACT FORMAT):
Frontend
Backend
Databases
Cloud & DevOps
Testing & Automation
AI/Automation Tools (if relevant)
Other Tools

Each category must have 12–20+ comma-separated skills, prioritizing JD keywords first. Follow the sample formatting and grouping rules as defined earlier.
9. Preserve all original quantified metrics (numbers, percentages, etc.) and actively introduce additional quantification in new or reworded bullets wherever possible. Use measurable outcomes, frequency, scope, or scale to increase the impact of each responsibility or accomplishment. Strive for at least 75% of all Experience bullet points to include a number, percentage, range, or scale to strengthen ATS, recruiter, and hiring manager perception.
10. Strictly maximize verb variety: No action verb (e.g., developed, led, built, designed, implemented, improved, created, managed, engineered, delivered, optimized, automated, collaborated, mentored) may appear more than twice in the entire document, and never in adjacent or back-to-back bullet points within or across jobs. Each bullet must start with a unique, action-oriented verb whenever possible.
11. In all Experience bullets, prefer keywords and phrasing directly from the JD where it truthfully reflects the candidate's background and would boost ATS/recruiter relevance.
12. Distribute JD-aligned technologies logically across roles.
- Assign primary/core technologies from the JD to the most recent or relevant positions.
- Assign secondary or supporting technologies to earlier roles.
- Ensure all key JD technologies appear at least once across the resume.

13. Ensure natural tone and realism. Only include technologies or responsibilities that the candidate could reasonably have used, based on their career path or industry.
14. The final resume should read as cohesive, naturally written, and contextually plausible—not artificially optimized.
15. Maintain all original section headers and formatting. Do not include commentary or extra text outside the resume.
Here is the base resume:
16. Include explicit database-related experience in the Professional Experience section.
17. Set the number of experiences in each company as 7-9 definitely and each experience must has 40-50 words.
${baseResume}

Here is the target job description:

${jobDescription}

Output the improved resume as plain text, exactly following the original resume's format—including the unchanged headline at the top. Clearly label sections (Summary, Professional Experience, Skills, Education, etc) with original spacing, section order, and no decorative lines or symbols.
  `;
}

// Helper to convert date format from MM/YYYY to MMM YYYY
function formatDate(dateStr: string): string {
  const monthNames = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
  ];

  // Handle different date formats
  if (dateStr.includes('–') || dateStr.includes('-')) {
    // Split by dash and format each part
    const parts = dateStr.split(/[–-]/).map(part => part.trim());
    return parts.map(part => {
      if (part.match(/^\d{2}\/\d{4}$/)) {
        const [month, year] = part.split('/');
        const monthIndex = parseInt(month) - 1;
        return `${monthNames[monthIndex]} ${year}`;
      }
      return part; // Return as-is if not in MM/YYYY format
    }).join(' – ');
  } else if (dateStr.match(/^\d{2}\/\d{4}$/)) {
    // Single date in MM/YYYY format
    const [month, year] = dateStr.split('/');
    const monthIndex = parseInt(month) - 1;
    return `${monthNames[monthIndex]} ${year}`;
  }

  return dateStr; // Return as-is if not in expected format
}

// Helper to wrap text within a max width
function wrapText(text: string, font: any, size: number, maxWidth: number) {
  const words = text.split(' ');
  const lines = [];
  let currentLine = '';
  for (let i = 0; i < words.length; i++) {
    const testLine = currentLine ? currentLine + ' ' + words[i] : words[i];
    const testWidth = font.widthOfTextAtSize(testLine, size);
    if (testWidth > maxWidth && currentLine) {
      lines.push(currentLine);
      currentLine = words[i];
    } else {
      currentLine = testLine;
    }
  }
  if (currentLine) lines.push(currentLine);
  return lines;
}

// Helper to draw text with bold segments (markdown **bold**)
function drawTextWithBold(
  page: any,
  text: string,
  x: number,
  y: number,
  font: any,
  fontBold: any,
  size: number,
  color: any
) {
  // Split by ** for bold
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  let offsetX = x;
  for (const part of parts) {
    if (part.startsWith('**') && part.endsWith('**')) {
      const content = part.slice(2, -2);
      page.drawText(content, { x: offsetX, y, size, font: fontBold, color });
      offsetX += fontBold.widthOfTextAtSize(content, size);
    } else {
      page.drawText(part, { x: offsetX, y, size, font, color });
      offsetX += font.widthOfTextAtSize(part, size);
    }
  }
}

// PDF generation function
async function generateResumePdf(resumeText: string): Promise<Uint8Array> {
  const { name, email, phone, location, linkedin, body } = parseResume(resumeText);

  console.log('resumeText', resumeText);
  const pdfDoc = await PDFDocument.create();
  let page = pdfDoc.addPage([595, 842]); // A4
  const font = await pdfDoc.embedFont(StandardFonts.Helvetica);
  const fontBold = await pdfDoc.embedFont(StandardFonts.HelveticaBold);

  // Color scheme for better visual hierarchy
  const BLACK = rgb(0, 0, 0);
  const DARK_BLUE = rgb(0.1, 0.2, 0.4); // For section headers
  const MEDIUM_BLUE = rgb(0.2, 0.4, 0.6); // For job titles
  const GRAY = rgb(0.4, 0.4, 0.4); // For company names and periods
  const DARK_GRAY = rgb(0.2, 0.2, 0.2); // For contact info

  const MARGIN_TOP = 72; // 1 inch = 72 points
  const MARGIN_BOTTOM = 50;
  const MARGIN_LEFT = 50;
  const MARGIN_RIGHT = 50;
  const PAGE_WIDTH = 595;
  const PAGE_HEIGHT = 842;
  const CONTENT_WIDTH = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT;

  // ATS-friendly font sizes
  const NAME_SIZE = 24; // Increased from 20 to 24 for better prominence
  const CONTACT_SIZE = 9; // Reduced from 10 to 9 (1px smaller)
  const SECTION_HEADER_SIZE = 14; // Increased from 12 to 14 for better visibility
  const BODY_SIZE = 11; // 10-12pt for body text

  // ATS-friendly line spacing (1.15 - 1.5 line height)
  const NAME_LINE_HEIGHT = NAME_SIZE * 0.8;
  const CONTACT_LINE_HEIGHT = CONTACT_SIZE * 1.5;
  const SECTION_LINE_HEIGHT = SECTION_HEADER_SIZE * 1.5;
  const BODY_LINE_HEIGHT = BODY_SIZE * 1.4;

  let y = PAGE_HEIGHT - MARGIN_TOP;
  const left = MARGIN_LEFT;
  const right = PAGE_WIDTH - MARGIN_RIGHT;

  // Name (large, bold, dark blue) - uppercase for emphasis
  if (name) {
    const nameLines = wrapText(name.toUpperCase(), fontBold, NAME_SIZE, CONTENT_WIDTH);
    for (const line of nameLines) {
      page.drawText(line, { x: left, y, size: NAME_SIZE, font: fontBold, color: DARK_BLUE });
      y -= NAME_LINE_HEIGHT;
    }
    y -= 2; // Small gap after name
  } else {
    y -= NAME_LINE_HEIGHT;
  }

  // Contact info on single line with bullet separators (like the image)
  const contactParts = [
    location,
    phone,
    email,
    linkedin
  ].filter(Boolean);

  if (contactParts.length > 0) {
    const contactLine = contactParts.join(' • ');
    const contactLines = wrapText(contactLine, font, CONTACT_SIZE, CONTENT_WIDTH);
    for (const line of contactLines) {
      page.drawText(line, { x: left, y, size: CONTACT_SIZE, font, color: DARK_GRAY });
      y -= CONTACT_LINE_HEIGHT;
    }
    y -= 4; // Small gap before horizontal line
  }

  // Draw horizontal line under contact info (like the image)
  page.drawLine({
    start: { x: left, y: y },
    end: { x: right, y: y },
    thickness: 1.5,
    color: DARK_BLUE
  });
  y -= 16; // Gap after horizontal line

  // Body (sections, skills, etc., wrapped)
  const bodyLines = body.split('\n');
  let inSkillsSection = false;
  const skills: string[] = [];
  for (let i = 0; i < bodyLines.length; i++) {
    const line = bodyLines[i].trim();
    if (!line) {
      y -= 6; // Reduced gap between paragraphs for ATS
      continue;
    }
    if (line.endsWith(':')) {
      y -= 12; // Increased gap before section header for better separation
      const sectionLines = wrapText(line, fontBold, SECTION_HEADER_SIZE, CONTENT_WIDTH);
      for (const sectionLine of sectionLines) {
        page.drawText(sectionLine, { x: left, y, size: SECTION_HEADER_SIZE, font: fontBold, color: DARK_BLUE });
        y -= SECTION_LINE_HEIGHT;
        if (y < MARGIN_BOTTOM) {
          page = pdfDoc.addPage([PAGE_WIDTH, PAGE_HEIGHT]);
          y = PAGE_HEIGHT - MARGIN_TOP;
        }
      }
      // Detect start of Skills section
      if (line.toLowerCase() === 'skills:') {
        inSkillsSection = true;
      }
    } else {
      // Check if this is a job experience line (Role at Company: Period)
      const isJobExperience = / at .+:.+/.test(line);

      if (isJobExperience) {
        // Parse job experience: Role at Company: Period
        const match = line.match(/^(.+?) at (.+?):\s*(.+)$/);
        if (match) {
          const [, jobTitle, companyName, period] = match;

          y -= 8; // Extra gap before job entry

          // Job Title (bold, blue)
          const titleLines = wrapText(jobTitle.trim(), fontBold, BODY_SIZE + 1, CONTENT_WIDTH - 10);
          for (const titleLine of titleLines) {
            page.drawText(titleLine, { x: left + 10, y, size: BODY_SIZE + 1, font: fontBold, color: MEDIUM_BLUE });
            y -= BODY_LINE_HEIGHT + 2;
            if (y < MARGIN_BOTTOM) {
              page = pdfDoc.addPage([PAGE_WIDTH, PAGE_HEIGHT]);
              y = PAGE_HEIGHT - MARGIN_TOP;
            }
          }

          // Company Name (italic, gray)
          const companyLines = wrapText(companyName.trim(), font, BODY_SIZE, CONTENT_WIDTH - 10);
          for (const companyLine of companyLines) {
            page.drawText(companyLine, { x: left + 10, y, size: BODY_SIZE, font, color: GRAY });
            y -= BODY_LINE_HEIGHT;
            if (y < MARGIN_BOTTOM) {
              page = pdfDoc.addPage([PAGE_WIDTH, PAGE_HEIGHT]);
              y = PAGE_HEIGHT - MARGIN_TOP;
            }
          }

          // Period (formatted and styled)
          const formattedPeriod = formatDate(period.trim());
          const periodLines = wrapText(formattedPeriod, font, BODY_SIZE - 1, CONTENT_WIDTH - 10);
          for (const periodLine of periodLines) {
            page.drawText(periodLine, { x: left + 10, y, size: BODY_SIZE - 1, font, color: GRAY });
            y -= BODY_LINE_HEIGHT - 2;
            if (y < MARGIN_BOTTOM) {
              page = pdfDoc.addPage([PAGE_WIDTH, PAGE_HEIGHT]);
              y = PAGE_HEIGHT - MARGIN_TOP;
            }
          }

          y -= 4; // Gap before experience bullets
        }
      } else {
        // Check if this is a skills category line (starts with ·)
        const isSkillsCategory = line.startsWith('·');

        if (isSkillsCategory) {
          // Skills category header (bold, dark blue)
          const categoryName = line.trim(); // Remove the · symbol
          const categoryLines = wrapText(categoryName, fontBold, BODY_SIZE + 1, CONTENT_WIDTH - 20);
          for (const categoryLine of categoryLines) {
            page.drawText(categoryLine, { x: left + 10, y, size: BODY_SIZE + 1, font: fontBold, color: MEDIUM_BLUE });
            y -= BODY_LINE_HEIGHT + 2;
            if (y < MARGIN_BOTTOM) {
              page = pdfDoc.addPage([PAGE_WIDTH, PAGE_HEIGHT]);
              y = PAGE_HEIGHT - MARGIN_TOP;
            }
          }
        } else {
          // Regular body text with proper indentation and line height
          const wrappedLines = wrapText(line, font, BODY_SIZE, CONTENT_WIDTH - 10); // indent body
          for (const wrappedLine of wrappedLines) {
            drawTextWithBold(page, wrappedLine, left + 10, y, font, fontBold, BODY_SIZE, BLACK);
            y -= BODY_LINE_HEIGHT;
            if (y < MARGIN_BOTTOM) {
              page = pdfDoc.addPage([PAGE_WIDTH, PAGE_HEIGHT]);
              y = PAGE_HEIGHT - MARGIN_TOP;
            }
          }
        }
      }
    }
    if (y < MARGIN_BOTTOM) {
      page = pdfDoc.addPage([PAGE_WIDTH, PAGE_HEIGHT]);
      y = PAGE_HEIGHT - MARGIN_TOP;
    }
  }
  // If the document ends while still in the skills section, render the skills
  if (inSkillsSection && skills.length > 0) {
    // Render comma-separated skills as wrapped text with better styling
    const skillsText = skills.join(' ');
    const wrappedSkillLines = wrapText(skillsText, font, BODY_SIZE, CONTENT_WIDTH - 20);
    for (const skillLine of wrappedSkillLines) {
      page.drawText(skillLine, { x: left + 20, y, size: BODY_SIZE, font, color: DARK_GRAY });
      y -= BODY_LINE_HEIGHT;
      if (y < MARGIN_BOTTOM) {
        page = pdfDoc.addPage([PAGE_WIDTH, PAGE_HEIGHT]);
        y = PAGE_HEIGHT - MARGIN_TOP;
      }
    }
  }

  return await pdfDoc.save();
}

export async function POST(req: NextRequest) {
  try {
    // 1. Parse form data
    const formData = await req.formData();
    const jobDescription = formData.get('job_description') as string;
    const company = formData.get('company') as string;
    const role = formData.get('role') as string;

    // Validate required fields
    if (!jobDescription || !company || !role) {
      return new NextResponse(
        JSON.stringify({ error: 'Missing required fields: job_description, company, role' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }

    // Check for OpenAI API key
    if (!process.env.OPENAI_API_KEY) {
      return new NextResponse(
        JSON.stringify({ error: 'OpenAI API key not configured' }),
        { status: 500, headers: { 'Content-Type': 'application/json' } }
      );
    }

    // 2. Load base resume from app directory (Vercel compatible)
    let baseResume: string = `
Senior Software Engineer

Lindsay Jackson
Senior Software Engineer
lindsay.jackson1630@gmail.com
+1 (325) 313 0928
Houston, TX, USA

Summary:

Senior Software Engineer with over 10 years of experience delivering scalable, high-performance web and mobile solutions across industries
including healthcare, finance, retail, and education. Proven expertise in cloud-native architectures, microservices, and full-stack development 
using technologies such as Python(Django, Flask), ASP.NET Core, React.js, Node.js, Docker, and Kubernetes. Adept at driving large-scale cloud migrations, real-time systems, and secure API integrations 
that comply with standards like HIPAA, FHIR, PCI DSS, and GDPR. Recognized for leading cross-functional teams and shipping mission-critical platforms 
used by millions of users globally, while optimizing system reliability, security, and user experience. Passionate about automation, DevOps (CI/CD), and building solutions 
that improve operational efficiency and end-user satisfaction.



Professional Experience:

Senior Software Engineer at Softeq Consulting: 01/2025 - Current
•	Architected a multi-agent orchestration platform in Python using LangChain and LlamaIndex, enabling reasoning-based workflows for triage, care planning, and EHR automation across hospital networks.
•	Built Retrieval-Augmented Generation (RAG) pipelines with Hugging Face Transformers, Pinecone vector DB, and AWS Bedrock, generating structured summaries from unstructured medical transcripts.
•	Developed knowledge graph integrations with Neo4j and AWS Neptune to unify patient history, claims data, and provider workflows, enabling downstream graph-based reasoning and predictive analytics.
•	Engineered real-time voice/NLP pipelines using PyTorch and Azure Speech Services, achieving sub-second latency for clinical dictation, diagnosis support, and sentiment-aware triage scoring.
•	Automated end-to-end MLOps workflows in AWS SageMaker for model training, evaluation, and deployment, with versioned model registries and drift detection.
•	Designed security guardrails for agentic systems (prompt injection detection, RBAC-aware inference policies, HIPAA-compliant audit logging).
•	Scaled multi-region AWS infrastructure with Kubernetes and Terraform, supporting tens of millions of inference requests/month while maintaining >99.99% uptime

Software Engineer(AI/ML) at AGKSOFT: 09/2023 – 01/2025
•	Designed an LLM-powered AI agent ecosystem for contextual pet care, using LangChain, OpenAI embeddings, and vectorized persistent memory across DynamoDB and Pinecone.
•	Built semantic knowledge retrieval pipelines for long-term memory and context preservation in multi-turn conversations with users and sitters.
•	Implemented real-time incident detection workflows (pet tracking, emergency alerts) with AWS EventBridge, SignalR, and speech-to-text services for hands-free caregiver updates.
•	Architected graph-based sitter matching algorithms using Neo4j and PyTorch GNNs, modeling relationships between pets, caregivers, and service history to predict sitter fit and trustworthiness.
•	Developed containerized AI microservices in Python and deployed via Kubernetes + AWS SageMaker, supporting A/B testing of new AI features at production scale.
•	Introduced a multi-tenant MLOps pipeline in GitLab CI/CD with rollout gating, enabling safe experiments in production AI services.
•	Mentored engineers on integrating agentic workflows beyond chatbots, focusing on multi-step reasoning and collaboration between agents.

Software Engineer(AI) at Oracle : 06/2020 – 09/2023
•	Built an image analysis engine in Python to detect anomalies and classify financial documents, applying probabilistic modeling to highlight errors or inconsistencies in scanned invoices and receipts.
•	Integrated Graph Databases (Neo4j) for visual relationship modeling between scanned documents, accounts, and transactions, uncovering patterns and dependencies.
•	Developed Generative AI image enhancement and inpainting agents using Hugging Face + TensorFlow to automatically clean, enhance, and reconstruct scanned documents and forms for audit-ready outputs.
•	Implemented AI-powered image assistants with AWS Rekognition + LangChain to guide users through document uploads, verification, and validation workflows, integrated directly into financial platforms.
•	Led ETL modernization efforts using dbt + Snowflake to preprocess, normalize, and store large-scale image datasets for ML model training and evaluation.
•	Built reinforcement learning agents for automated document categorization and prioritization, dynamically adjusting based on document quality, type, and processing urgency.
•	Deployed image processing AI workflows into AWS SageMaker pipelines, scaling predictive and classification models to millions of scanned documents across enterprise platforms.

AI/ML Developer I at The Home Spot : 02/2017 – 06/2020
•	Built a real-time inventory forecasting agent using TensorFlow, Kafka Streams, and Snowflake, predicting stockouts and automatically triggering supplier reorders.
•	Designed a knowledge graph of products, suppliers, and store locations in AWS Neptune, applying GNNs to optimize distribution during seasonal surges.
•	Developed a graph-powered recommendation engine for product bundling, trained on customer purchase histories and store-level sales graphs.
•	Deployed predictive AI workflows on AWS SageMaker + Kubernetes, reducing per-store out-of-stock incidents by 28%.
•	Created containerized ETL pipelines in Airflow to unify transactional, IoT, and vendor data into Snowflake for downstream AI workloads.
•	Integrated symbolic reasoning rules into replenishment workflows, combining deterministic business constraints with ML predictions for explainable decision-making.
•	Automated CI/CD pipelines for AI workloads with Jenkins + GitLab CI, enabling daily retraining and deployment of high-volume predictive models.

AI Focus Web Developer at Amazon : 11/2015 – 02/2017
•	Built a collaborative learning agent in Python and Node.js using reinforcement learning to adapt quizzes dynamically to student performance.
•	Implemented a graph-based content recommendation engine with Neo4j, modeling relationships between concepts, student progress, and instructor materials.
•	Developed a real-time classroom orchestration system using WebSockets + LangChain to manage student/teacher multi-agent interactions.
•	Integrated speech-based reasoning agents with Web Speech API + Hugging Face for accessibility and adaptive test assistance.
•	Deployed scalable AI-driven dashboards (React.js + D3.js) visualizing sentiment, progress, and content mastery, powered by predictive ML services in AWS.
•	Designed automated knowledge tracing pipelines in TensorFlow to predict dropout risk and recommend intervention paths for instructors.
•	Collaborated with researchers on symbolic AI integrations into student-facing tools, enhancing explainability in adaptive learning recommendations.


Skills:

LLMs (GPT, Claude, LLaMA)
LangChain
Hugging Face Transformers
RAG
Vector Databases (Pinecone, Weaviate, FAISS)
PyTorch
TensorFlow
Scikit-learn
OpenCV
YOLO
NLP
CV
Recommendation Engines
MLflow
Kubeflow
Weights & Biases
DVC
Databricks
Spark
Kafka
Airflow
Docker
Kubernetes
Terraform
AWS SageMaker
Bedrock
Azure ML
GCP Vertex AI
Anthropic API
OpenAI API
Python (FastAPI, Flask, Django)
Node.js (Express, Nest)
C#
ASP.NET Core
React.js
React Native
TypeScript
PostgreSQL
MongoDB
Redis
Snowflake
BigQuery
Vector DBs
HIPAA
PCI DSS
GDPR
CCPA
Differential Privacy
Federated Learning Security
ELK Stack
Prometheus
Grafana
PyTest
Postman
Cypress

Education:
Bachelor of Science, Computer Science (10/2011 – 08/2015)
The University of Tokyo | Japan
    `;
    // 3. Tailor resume with OpenAI
    const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    const prompt = buildPrompt(baseResume, jobDescription);

    const completion = await openai.chat.completions.create({
      model: process.env.OPENAI_VERSION || 'gpt-3.5-turbo',
      messages: [
        { role: 'system', content: 'You are a helpful assistant for creating professional resume content.' },
        { role: 'user', content: prompt }
      ],
      temperature: 0.7,
      max_tokens: 4096
    });

    const tailoredResume = completion.choices[0].message.content || '';

    if (!tailoredResume) {
      return new NextResponse(
        JSON.stringify({ error: 'Failed to generate tailored resume content' }),
        { status: 500, headers: { 'Content-Type': 'application/json' } }
      );
    }

    // 4. Generate PDF
    const pdfBytes = await generateResumePdf(tailoredResume);

    // 5. Return PDF as response
    const fileBase = `Lindsay_Jackson_${company.replace(/[^a-zA-Z0-9_]/g, '_')}_${role.replace(/[^a-zA-Z0-9_]/g, '_')}`;
    return new NextResponse(Buffer.from(pdfBytes), {
      status: 200,
      headers: {
        'Content-Type': 'application/pdf',
        'Content-Disposition': `attachment; filename="${fileBase}.pdf"`
      }
    });
  } catch (error) {
    console.error('Error generating PDF:', error);
    return new NextResponse(
      JSON.stringify({
        error: 'Internal server error',
        details: error instanceof Error ? error.message : 'Unknown error'
      }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}