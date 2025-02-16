# app.R

library(shiny)
library(ggplot2)
library(dplyr)
library(readr)
library(lubridate)

# ---------------------------
# Data & Helper Functions
# ---------------------------

# Regression model coefficients and residual SDs
regression_models <- list(
  CVLT_totaldeacertos = list(constant = 8.512324, age = -0.14798, age2 = 0.001373,
                             sex = 0.176426, education = 0.364315, residual_sd = 2.527166),
  BVMT_Total = list(constant = 11.58455, age = -0.14752, age2 = 0.000896,
                    sex = -0.19042, education = 0.22895, residual_sd = 2.626665),
  SDMT = list(constant = 9.248778, age = -0.01094, age2 = -0.00086,
              sex = -0.4714, education = 0.263055, residual_sd = 2.48323)
)

# Conversion tables for raw score → scaled score
conversion_table <- list(
  CVLT_totaldeacertos = list(
    "1" = c(-Inf, 19), "2" = c(20, 28), "3" = c(29, 31), "4" = c(32, 35),
    "5" = c(36, 39), "6" = c(40, 41), "7" = c(42, 44), "8" = c(45, 48),
    "9" = c(49, 52), "10" = c(53, 56), "11" = c(57, 60), "12" = c(61, 64),
    "13" = c(65, 66), "14" = c(67, 69), "15" = c(70, 71), "16" = c(72, 72),
    "17" = c(73, 74), "18" = c(75, 75), "19" = c(76, Inf)
  ),
  BVMT_Total = list(
    "1" = c(-Inf, 2), "2" = c(3, 5), "3" = c(6, 8), "4" = c(9, 12),
    "5" = c(13, 17), "6" = c(18, 20), "7" = c(21, 23), "8" = c(24, 26),
    "9" = c(27, 28), "10" = c(29, 30), "11" = c(31, 32), "12" = c(33, 34),
    "13" = c(35, 35), "14" = c(36, 36)
  ),
  SDMT = list(
    "1" = c(-Inf, 9), "2" = c(10, 17), "3" = c(18, 23), "4" = c(24, 29),
    "5" = c(30, 36), "6" = c(37, 43), "7" = c(44, 49), "8" = c(50, 53),
    "9" = c(54, 58), "10" = c(59, 62), "11" = c(63, 68), "12" = c(69, 74),
    "13" = c(75, 79), "14" = c(80, 93), "15" = c(94, 107), "16" = c(108, Inf)
  )
)

# Function to convert a raw score into a scaled score
convert_to_scaled_score <- function(raw_score, measure) {
  table <- conversion_table[[measure]]
  for (score in names(table)) {
    limits <- table[[score]]
    if (raw_score >= limits[1] && raw_score <= limits[2]) {
      return(as.numeric(score))
    }
  }
  return(NA)
}

# Calculate the predicted scaled score (PSS)
calculate_predicted_scaled_score <- function(age, sex, education, measure) {
  model <- regression_models[[measure]]
  age2 <- age^2
  sex_val <- ifelse(sex == "Masculino", 1, 2)
  pss <- model$constant + model$age * age + model$age2 * age2 +
    model$sex * sex_val + model$education * education
  return(pss)
}

# Interpret the percentile to obtain classification and a representative color
interpret_percentile <- function(percentile) {
  if (percentile >= 98) {
    return(list(classification = "Excepcionalmente Alto", color = "#00008B"))
  } else if (percentile >= 90) {
    return(list(classification = "Acima da Média", color = "#0000FF"))
  } else if (percentile >= 75) {
    return(list(classification = "Médio-Alto", color = "#00FFFF"))
  } else if (percentile >= 25) {
    return(list(classification = "Médio", color = "#00FF00"))
  } else if (percentile >= 9) {
    return(list(classification = "Médio-Baixo", color = "#FFD700"))
  } else if (percentile >= 2) {
    return(list(classification = "Abaixo da Média", color = "#FF4500"))
  } else {
    return(list(classification = "Excepcionalmente Baixo", color = "#FF0000"))
  }
}

# Plot the standard normal distribution with the computed z-score highlighted
plot_normal_distribution <- function(z_score, measure_name, percentile, classification, color) {
  x_vals <- seq(-4, 4, length.out = 100)
  df <- data.frame(x = x_vals, y = dnorm(x_vals))
  point_y <- dnorm(z_score)
  
  p <- ggplot(df, aes(x = x, y = y)) +
    geom_line() +
    geom_point(aes(x = z_score, y = point_y), color = color, size = 3) +
    geom_text(aes(x = z_score, y = point_y,
                  label = paste0("Z = ", round(z_score,2),
                                 "\nP = ", round(percentile,1), "%",
                                 "\n", classification)),
              vjust = -1, size = 3) +
    labs(title = paste("Valores normativos para", measure_name),
         x = "Z-score", y = "Densidade de Probabilidade") +
    theme_minimal()
  return(p)
}

# Function to generate a CSV template as a data frame
generate_csv_template <- function() {
  template <- data.frame(
    PatientName = character(0),
    Sex = c("Masculino", "Feminino"),
    Age = integer(0),
    Education = integer(0),
    TestDate = as.Date(character(0)),
    SDMT_Raw = numeric(0),
    CVLT_Raw = numeric(0),
    BVMT_Raw = numeric(0)
  )
  # For a template file, you might include one example row:
  template <- rbind(template, data.frame(
    PatientName = "Exemplo",
    Sex = "Masculino",
    Age = 40,
    Education = 12,
    TestDate = Sys.Date(),
    SDMT_Raw = 60,
    CVLT_Raw = 50,
    BVMT_Raw = 20
  ))
  return(template)
}

# ---------------------------
# UI
# ---------------------------
ui <- navbarPage("Calculadora Normativa do BICAMS",
                 
  tabPanel("Avaliação Individual",
           sidebarLayout(
             sidebarPanel(
               textInput("patient_name", "Nome ou Código do Paciente", value = "Paciente Exemplo"),
               selectInput("sex", "Sexo", choices = c("Masculino", "Feminino")),
               sliderInput("age", "Idade (anos)", min = 18, max = 100, value = 40),
               sliderInput("education", "Escolaridade (anos)", min = 1, max = 20, value = 12),
               dateInput("test_date", "Data do Teste", value = Sys.Date()),
               hr(),
               h4("SDMT"),
               checkboxInput("sdmt_na", "Não se aplica", value = FALSE),
               conditionalPanel(
                 condition = "input.sdmt_na == false",
                 radioButtons("sdmt_input_method", "Como deseja inserir a pontuação?",
                              choices = c("Deslizar", "Digite"), selected = "Deslizar"),
                 conditionalPanel(
                   condition = "input.sdmt_input_method == 'Deslizar'",
                   sliderInput("sdmt_raw_slider", "Pontuação SDMT", min = 0, max = 120, value = 60)
                 ),
                 conditionalPanel(
                   condition = "input.sdmt_input_method == 'Digite'",
                   numericInput("sdmt_raw_num", "Pontuação SDMT", value = 60, min = 0, max = 120)
                 )
               ),
               hr(),
               h4("CVLT-II"),
               checkboxInput("cvlt_na", "Não se aplica", value = FALSE),
               conditionalPanel(
                 condition = "input.cvlt_na == false",
                 radioButtons("cvlt_input_method", "Como deseja inserir a pontuação?",
                              choices = c("Deslizar", "Digite"), selected = "Deslizar"),
                 conditionalPanel(
                   condition = "input.cvlt_input_method == 'Deslizar'",
                   sliderInput("cvlt_raw_slider", "Pontuação Total CVLT", min = 0, max = 80, value = 50)
                 ),
                 conditionalPanel(
                   condition = "input.cvlt_input_method == 'Digite'",
                   numericInput("cvlt_raw_num", "Pontuação Total CVLT", value = 50, min = 0, max = 80)
                 )
               ),
               hr(),
               h4("BVMT-R"),
               checkboxInput("bvmt_na", "Não se aplica", value = FALSE),
               conditionalPanel(
                 condition = "input.bvmt_na == false",
                 radioButtons("bvmt_input_method", "Como deseja inserir a pontuação?",
                              choices = c("Deslizar", "Digite"), selected = "Deslizar"),
                 conditionalPanel(
                   condition = "input.bvmt_input_method == 'Deslizar'",
                   sliderInput("bvmt_raw_slider", "Pontuação Total BVMT", min = 0, max = 36, value = 20)
                 ),
                 conditionalPanel(
                   condition = "input.bvmt_input_method == 'Digite'",
                   numericInput("bvmt_raw_num", "Pontuação Total BVMT", value = 20, min = 0, max = 36)
                 )
               )
             ),
             
             mainPanel(
               h3("Resultados da Avaliação"),
               uiOutput("individual_results")
             )
           )
  ),
  
  tabPanel("Processamento em Lote",
           sidebarLayout(
             sidebarPanel(
               h4("Download do Template CSV"),
               downloadButton("downloadTemplate", "Baixar Template CSV"),
               hr(),
               h4("Upload do CSV Preenchido"),
               fileInput("csv_upload", "Escolha o arquivo CSV", accept = ".csv"),
               hr(),
               downloadButton("downloadResults", "Baixar Resultados Processados")
             ),
             mainPanel(
               h3("Resultados em Lote"),
               tableOutput("batch_table")
             )
           )
  )
)

# ---------------------------
# Server
# ---------------------------
server <- function(input, output, session) {
  
  # ---------------------------
  # Individual Assessment Calculations
  # ---------------------------
  
  # Helper to choose the raw value (slider or number input)
  get_raw_value <- function(methodInput, sliderInput, numInput) {
    if(methodInput == "Deslizar") {
      return(sliderInput)
    } else {
      return(numInput)
    }
  }
  
  individual_results <- reactive({
    req(input$patient_name, input$sex, input$age, input$education, input$test_date)
    res <- list()
    
    # SDMT
    if (!input$sdmt_na) {
      sdmt_raw <- get_raw_value(input$sdmt_input_method, input$sdmt_raw_slider, input$sdmt_raw_num)
      sdmt_scaled <- convert_to_scaled_score(sdmt_raw, "SDMT")
      if (!is.na(sdmt_scaled)) {
        sdmt_pss <- calculate_predicted_scaled_score(input$age, input$sex, input$education, "SDMT")
        sdmt_z <- (sdmt_scaled - sdmt_pss) / regression_models$SDMT$residual_sd
        sdmt_percentile <- pnorm(sdmt_z) * 100
        interp <- interpret_percentile(sdmt_percentile)
        res$SDMT <- list(raw = sdmt_raw, scaled = sdmt_scaled, pss = sdmt_pss,
                         z = sdmt_z, percentile = sdmt_percentile,
                         classification = interp$classification,
                         plot = plot_normal_distribution(sdmt_z, "SDMT", sdmt_percentile,
                                                         interp$classification, interp$color))
      }
    }
    
    # CVLT-II
    if (!input$cvlt_na) {
      cvlt_raw <- get_raw_value(input$cvlt_input_method, input$cvlt_raw_slider, input$cvlt_raw_num)
      cvlt_scaled <- convert_to_scaled_score(cvlt_raw, "CVLT_totaldeacertos")
      if (!is.na(cvlt_scaled)) {
        cvlt_pss <- calculate_predicted_scaled_score(input$age, input$sex, input$education, "CVLT_totaldeacertos")
        cvlt_z <- (cvlt_scaled - cvlt_pss) / regression_models$CVLT_totaldeacertos$residual_sd
        cvlt_percentile <- pnorm(cvlt_z) * 100
        interp <- interpret_percentile(cvlt_percentile)
        res$CVLT <- list(raw = cvlt_raw, scaled = cvlt_scaled, pss = cvlt_pss,
                         z = cvlt_z, percentile = cvlt_percentile,
                         classification = interp$classification,
                         plot = plot_normal_distribution(cvlt_z, "CVLT-II", cvlt_percentile,
                                                         interp$classification, interp$color))
      }
    }
    
    # BVMT-R
    if (!input$bvmt_na) {
      bvmt_raw <- get_raw_value(input$bvmt_input_method, input$bvmt_raw_slider, input$bvmt_raw_num)
      bvmt_scaled <- convert_to_scaled_score(bvmt_raw, "BVMT_Total")
      if (!is.na(bvmt_scaled)) {
        bvmt_pss <- calculate_predicted_scaled_score(input$age, input$sex, input$education, "BVMT_Total")
        bvmt_z <- (bvmt_scaled - bvmt_pss) / regression_models$BVMT_Total$residual_sd
        bvmt_percentile <- pnorm(bvmt_z) * 100
        interp <- interpret_percentile(bvmt_percentile)
        res$BVMT <- list(raw = bvmt_raw, scaled = bvmt_scaled, pss = bvmt_pss,
                         z = bvmt_z, percentile = bvmt_percentile,
                         classification = interp$classification,
                         plot = plot_normal_distribution(bvmt_z, "BVMT-R", bvmt_percentile,
                                                         interp$classification, interp$color))
      }
    }
    return(res)
  })
  
  output$individual_results <- renderUI({
    res <- individual_results()
    output_list <- list()
    
    if (!is.null(res$SDMT)) {
      output_list <- c(output_list,
                       h4("SDMT"),
                       verbatimTextOutput("sdmt_text"),
                       plotOutput("sdmt_plot"))
    }
    if (!is.null(res$CVLT)) {
      output_list <- c(output_list,
                       h4("CVLT-II"),
                       verbatimTextOutput("cvlt_text"),
                       plotOutput("cvlt_plot"))
    }
    if (!is.null(res$BVMT)) {
      output_list <- c(output_list,
                       h4("BVMT-R"),
                       verbatimTextOutput("bvmt_text"),
                       plotOutput("bvmt_plot"))
    }
    if (length(output_list) == 0) {
      output_list <- h4("Nenhum teste foi processado.")
    }
    do.call(tagList, output_list)
  })
  
  output$sdmt_text <- renderPrint({
    res <- individual_results()$SDMT
    if (!is.null(res)) {
      cat("Raw Score:", res$raw, "\n")
      cat("Scaled Score:", res$scaled, "\n")
      cat("PSS:", round(res$pss, 2), "\n")
      cat("Z-score:", round(res$z, 2), "\n")
      cat("Percentil:", round(res$percentile, 1), "%\n")
      cat("Classificação:", res$classification, "\n")
    }
  })
  output$cvlt_text <- renderPrint({
    res <- individual_results()$CVLT
    if (!is.null(res)) {
      cat("Raw Score:", res$raw, "\n")
      cat("Scaled Score:", res$scaled, "\n")
      cat("PSS:", round(res$pss, 2), "\n")
      cat("Z-score:", round(res$z, 2), "\n")
      cat("Percentil:", round(res$percentile, 1), "%\n")
      cat("Classificação:", res$classification, "\n")
    }
  })
  output$bvmt_text <- renderPrint({
    res <- individual_results()$BVMT
    if (!is.null(res)) {
      cat("Raw Score:", res$raw, "\n")
      cat("Scaled Score:", res$scaled, "\n")
      cat("PSS:", round(res$pss, 2), "\n")
      cat("Z-score:", round(res$z, 2), "\n")
      cat("Percentil:", round(res$percentile, 1), "%\n")
      cat("Classificação:", res$classification, "\n")
    }
  })
  
  output$sdmt_plot <- renderPlot({
    res <- individual_results()$SDMT
    if (!is.null(res)) {
      print(res$plot)
    }
  })
  output$cvlt_plot <- renderPlot({
    res <- individual_results()$CVLT
    if (!is.null(res)) {
      print(res$plot)
    }
  })
  output$bvmt_plot <- renderPlot({
    res <- individual_results()$BVMT
    if (!is.null(res)) {
      print(res$plot)
    }
  })
  
  # ---------------------------
  # Batch Processing (CSV Template & Upload)
  # ---------------------------
  
  # Download the CSV template
  output$downloadTemplate <- downloadHandler(
    filename = function() {
      paste0("BICAMS_Template_", Sys.Date(), ".csv")
    },
    content = function(file) {
      write_csv(generate_csv_template(), file)
    }
  )
  
  # Reactive: Process the uploaded CSV file
  processed_data <- reactive({
    req(input$csv_upload)
    df <- read_csv(input$csv_upload$datapath, col_types = cols(
      PatientName = col_character(),
      Sex = col_character(),
      Age = col_double(),
      Education = col_double(),
      TestDate = col_date(),
      SDMT_Raw = col_double(),
      CVLT_Raw = col_double(),
      BVMT_Raw = col_double()
    ))
    
    # For each row, calculate scores for available tests.
    df <- df %>%
      rowwise() %>%
      mutate(
        SDMT_Scaled = ifelse(!is.na(SDMT_Raw),
                             convert_to_scaled_score(SDMT_Raw, "SDMT"), NA),
        SDMT_PSS = ifelse(!is.na(SDMT_Scaled),
                          calculate_predicted_scaled_score(Age, Sex, Education, "SDMT"), NA),
        SDMT_Z = ifelse(!is.na(SDMT_PSS),
                        (SDMT_Scaled - SDMT_PSS) / regression_models$SDMT$residual_sd, NA),
        SDMT_Percentile = ifelse(!is.na(SDMT_Z),
                                 pnorm(SDMT_Z) * 100, NA),
        SDMT_Classification = ifelse(!is.na(SDMT_Percentile),
                                     interpret_percentile(SDMT_Percentile)$classification, NA),
        CVLT_Scaled = ifelse(!is.na(CVLT_Raw),
                             convert_to_scaled_score(CVLT_Raw, "CVLT_totaldeacertos"), NA),
        CVLT_PSS = ifelse(!is.na(CVLT_Scaled),
                          calculate_predicted_scaled_score(Age, Sex, Education, "CVLT_totaldeacertos"), NA),
        CVLT_Z = ifelse(!is.na(CVLT_PSS),
                        (CVLT_Scaled - CVLT_PSS) / regression_models$CVLT_totaldeacertos$residual_sd, NA),
        CVLT_Percentile = ifelse(!is.na(CVLT_Z),
                                 pnorm(CVLT_Z) * 100, NA),
        CVLT_Classification = ifelse(!is.na(CVLT_Percentile),
                                     interpret_percentile(CVLT_Percentile)$classification, NA),
        BVMT_Scaled = ifelse(!is.na(BVMT_Raw),
                             convert_to_scaled_score(BVMT_Raw, "BVMT_Total"), NA),
        BVMT_PSS = ifelse(!is.na(BVMT_Scaled),
                          calculate_predicted_scaled_score(Age, Sex, Education, "BVMT_Total"), NA),
        BVMT_Z = ifelse(!is.na(BVMT_PSS),
                        (BVMT_Scaled - BVMT_PSS) / regression_models$BVMT_Total$residual_sd, NA),
        BVMT_Percentile = ifelse(!is.na(BVMT_Z),
                                 pnorm(BVMT_Z) * 100, NA),
        BVMT_Classification = ifelse(!is.na(BVMT_Percentile),
                                     interpret_percentile(BVMT_Percentile)$classification, NA)
      ) %>%
      ungroup()
    return(df)
  })
  
  output$batch_table <- renderTable({
    req(processed_data())
    processed_data()
  }, striped = TRUE, hover = TRUE)
  
  # Allow downloading the processed CSV
  output$downloadResults <- downloadHandler(
    filename = function() {
      paste0("BICAMS_Results_", Sys.Date(), ".csv")
    },
    content = function(file) {
      write_csv(processed_data(), file)
    }
  )
  
}

# ---------------------------
# Run the App
# ---------------------------
shinyApp(ui, server)
