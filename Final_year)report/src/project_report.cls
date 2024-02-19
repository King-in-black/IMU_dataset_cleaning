\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{project_report}[2021/12/02 (oh209)]

% Passes and class options to the underlying article class
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions
\LoadClass[twoside]{article}


% Font settings
%\renewcommand{\rmdefault}{mathptmx}

%\newcommand{\serifnormal}{\fontspec{TeX Gyre Termes}}

% Page dimensions
\RequirePackage[paperwidth=210mm, paperheight=297mm, left=25mm, right=25mm, top=27.5mm, bottom=32mm, foot=8mm, asymmetric]{geometry}
%\setlength{\columnsep}{5.2mm}
\setlength{\parindent}{5.2mm}

% Maths
\RequirePackage{amsmath}
\RequirePackage{amsthm}

% Cover page
\newcommand{\@supervisor}{}
\newcommand{\supervisor}[1]{\renewcommand{\@supervisor}{Supervisor: #1}}
%\newcommand{\@advisor}{}
%\newcommand{\advisor}[1]{\renewcommand{\@advisor}{Advisor: #1}}

\def\college#1{\gdef\@college{#1}}
\newcommand{\coverpage}{%
%	\newgeometry{hmarginratio=1:1, margin=1.5in}
	\thispagestyle{empty}
	\onecolumn{
	\begin{center}
	\vspace*{3\topmargin}
	\noindent\makebox[\textwidth]{
		\includegraphics[width=1.0\paperwidth]{darkblue_banner}
	}
	\end{center}	
	\begin{center}

		\vspace*{3cm}

		\parbox[t][4cm]{\textwidth}{
%		\begin{minipage}{\textwidth}	
			\begin{center}
			\addtolength{\baselineskip}{0.5\baselineskip}
			{\huge\textsc \@title} %fontsize{22pt}{24pt}\selectfont0
	
			\vspace{2em}
	
			{\LARGE \@author} %\fontsize{18pt}{20pt}\selectfont

			\end{center}
		}
%		\end{minipage}
		\vspace{\fill}
		

		\vspace{3em}
		
		\parbox[t]{\textwidth}{

			\begin{center}

			{\Large\bfseries 3\textsuperscript{rd} Year Project Interim Report} %\fontsize{16pt}{18pt}\selectfont
		
			\vspace{1.5em}
			
			{\Large Department of Electronic \&\\\vspace*{5pt}Electrical Engineering} %\fontsize{16pt}{18pt}\selectfont
	
			\vspace{1.5em}
	
			{\Large UCL} %\fontsize{16pt}{18pt}\selectfont
	
			\vspace{2em}
	
			{\large \@supervisor}\\\vspace*{5pt}
 			%{\large \@advisor}
			%{\large \@college\hfill \hfill \@date} %\fontsize{14pt}{16pt}\selectfont

			\vspace{5em}
			
			{\large \@date}
			
			\end{center}	
			
			\vspace*{10em}
		}

	\end{center}
	}	
	\clearpage
}

% Declaration
\newcommand{\@aiuse}{}
\newcommand{\aiuse}[1]{\renewcommand{\@aiuse}{#1}}

\def\wordcount#1{\gdef\@wordcount{#1 words}}
\newcommand{\declaration}{%
	\thispagestyle{empty}
	{\large
	\parindent0pt
	\setlength{\parskip}{20pt}
	\noindent I have read and understood UCL's and the Department's statements and guidelines concerning plagiarism.
	
	I declare that all material described in this report is my own work except where explicitly and individually indicated in the text. This includes ideas described in the text, figures and computer programs. 
	
	\ifx\aiuse\@empty\else
	\@aiuse
	\fi
	
	This report contains \pageref{LastMainPage} pages (excluding this page and the appendices) and \@wordcount.

	\vspace{2em}
	
	\noindent Signed:
	\begin{minipage}[t]{15em}
		\centering
		\rule{\textwidth}{0.5pt} \newline
		% replace the line above with 
		%\rule{\textwidth}{0.5pt} \hspace{-14em}\rule{0pt}{-3ex} \includegraphics[width=13em]{signature}
		\newline
		(Student)
	\end{minipage}
	\hspace{\fill} Date: \rule{9em}{0.5pt} \newline

	\vspace{\fill}

	\clearpage
	\setcounter{page}{1}	% let's start counting the pages from the first page of the actual report
	}
}

% Title
\RequirePackage{titling}
\renewcommand{\maketitle}{
		\begin{raggedright}
			% Title
			{\sffamily\LARGE\textbf{\@title}} %\fontsize{22pt}{24pt}\selectfont  

			\vspace*{10pt}
			% Author
			{\sffamily\theauthor} %\fontsize{9.7pt}{12pt}\selectfont 

			\vspace*{1em}

		\end{raggedright}
}

% Section headers
\RequirePackage{titlesec}
\titleformat{\section}{\sffamily\bfseries\normalsize}{\arabic{section}}{0.5em}{} % \fontsize{7.9pt}{8.5pt}
\titlespacing*{\section}{0pt}{1em}{0pt}
\titleformat{\subsection}{\sffamily\bfseries\normalsize}{\arabic{section}.\arabic{subsection}}{0.5em}{}
\titlespacing*{\subsection}{0pt}{1em}{0pt}
%\titleformat{\subsubsection}[runin]{\sffamily\bfseries\itseries}{\arabic{subsubsection}}{0em}{}[.]
\titleformat{\subsubsection}{\sffamily\bfseries\itshape}{\arabic{section}.\arabic{subsection}.\arabic{subsubsection}}{0.5em}{}
\titlespacing*{\subsubsection}{0pt}{1em}{0pt}

% Keywords
\newenvironment{keywords}{%
	\sffamily\bfseries\small\setlength{\parindent}{0pt}Keywords:
}{\par\vspace*{1em}}

%\newcommand{\@keywords}{}
%\newcommand{\keywords}[1]{\renewcommand{\@keywords}{Keywords: #1}}
%\newcommand{\keywords}[1]{\sffamily\bfseries\small\setlength{\parindent}{0pt} Keywords: #1}
%{\large \@keywords}


% Abstract
\renewenvironment{abstract}{%
	\sffamily\bfseries\small\setlength{\parindent}{0pt}
}{\par\vspace*{1em}}

% Table of contents
\let\oldtableofcontents\tableofcontents
\renewcommand{\tableofcontents}{\sffamily\oldtableofcontents\vspace*{2em}\rmfamily} % after the table of contents  we need to switch back to serif font

% Figure and table captions
\RequirePackage{caption}
\DeclareCaptionFormat{sffamily}{\sffamily\normalsize\textbf{#1#2}#3} %\fontsize{7.9pt}{8.5pt}\selectfont 

\captionsetup[figure]{format=sffamily}

\captionsetup[table]{format=sffamily}
\renewenvironment{table}{%
	\renewcommand* {\@floatboxreset}{%
		\sffamily\normalsize\@setminipage}% \fontsize{7.9pt}{8.5pt}\selectfont
	\@float{table}%
}{%
	\end@float
}

% Reduce itemseparation in lists
%\let\olditem\item
%\renewcommand{\item}{%
%\olditem\vspace{-10pt}}

% Footer
\RequirePackage{fancyhdr}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyfoot[LO,RE]{\sffamily\normalsize \@author } %\fontsize{7.9pt}{8.5pt}\selectfont
\fancyfoot[RO]{\sffamily\normalsize%
	 UCL EEE\hspace*{3pt}$\vert$\hspace*{3pt}3\textsuperscript{rd} year project final report\hspace*{3pt}$\vert$\hspace*{5pt}\thepage}
\fancyfoot[LE]{\sffamily\normalsize%
	 \thepage\hspace*{5pt}$\vert$\hspace*{3pt}UCL EEE\hspace*{3pt}$\vert$\hspace*{3pt}3\textsuperscript{rd} year project final report}
\pagestyle{fancy}


% Appendices
\let\oldappendix\appendix
\renewcommand{\appendix}{
\label{LastMainPage}
\newpage
\onecolumn
%\pagestyle{empty}
\titleformat{\section}{\sffamily\bfseries\normalsize}{\Alph{section}}{0.5em}{} %\fontsize{7.9pt}{8.5pt}
\titlespacing*{\section}{0pt}{1em}{0pt}

{\noindent\Large\sffamily\textbf{Appendices}} \newline
\oldappendix
}

% Alter some LaTeX defaults for better treatment of figures
\renewcommand{\topfraction}{0.999}
\renewcommand{\bottomfraction}{0.999}
\setcounter{topnumber}{2}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{4}
\setcounter{dbltopnumber}{2}
\renewcommand{\dbltopfraction}{0.999}
\renewcommand{\textfraction}{0.01}
\renewcommand{\floatpagefraction}{0.99}
\renewcommand{\dblfloatpagefraction}{0.99}